"""Audit fresh DROID shards and freeze a group-safe next-action task."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from scripts.audit_droid100_groups import (  # noqa: E402
    episode_example_context,
    episode_example_feature_counts,
    episode_example_float_values,
    episode_example_instructions,
    iter_tfrecord,
)
from scripts.build_droid_group_split import build_group_split  # noqa: E402
from scripts.download_droid_fresh_cohort import validate_manifest  # noqa: E402
from tiny_omni_decision.droid_fresh_next_action import (  # noqa: E402
    build_task_manifest,
    canonical_path,
    label_from_velocity,
    stable_id,
    uniform_indices,
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _audit(data_dir: Path, manifest: dict[str, Any], manifest_sha: str) -> dict[str, Any]:
    data_root = data_dir / manifest["prefix"]
    info_raw = (data_root / "dataset_info.json").read_bytes()
    if sha256(info_raw) != manifest["dataset_info_sha256"]:
        raise ValueError("dataset_info.json SHA-256 mismatch")
    features_raw = (data_root / "features.json").read_bytes()
    if sha256(features_raw) != manifest["features_sha256"]:
        raise ValueError("features.json SHA-256 mismatch")
    info = json.loads(info_raw)
    train = next((row for row in info["splits"] if row.get("name") == "train"), None)
    if train is None or len(train.get("shardLengths", [])) != 2048:
        raise ValueError("metadata does not describe the pinned DROID train split")
    episodes = []
    source_files: set[str] = set()
    recording_groups: set[str] = set()
    instructions: Counter[str] = Counter()
    shard_counts: dict[str, int] = {}
    camera_keys = (
        "steps/observation/exterior_image_1_left",
        "steps/observation/exterior_image_2_left",
        "steps/observation/wrist_image_left",
    )
    sensor_widths = {
        "steps/observation/joint_position": 7,
        "steps/observation/cartesian_position": 6,
        "steps/observation/gripper_position": 1,
    }
    for shard in manifest["selected_shards"]:
        index = int(shard["index"])
        if int(train["shardLengths"][index]) != int(shard["episodes"]):
            raise ValueError(f"dataset_info count mismatch for shard {index}")
        path = data_root / Path(shard["object_name"]).name
        rows = []
        for record in iter_tfrecord(path):
            context = episode_example_context(record)
            file_path = canonical_path(context["episode_metadata/file_path"])
            group_path = canonical_path(context["episode_metadata/recording_folderpath"])
            if file_path in source_files:
                raise ValueError("duplicate episode source identity across fresh shards")
            source_files.add(file_path)
            recording_groups.add(group_path)
            counts = episode_example_feature_counts(record)
            steps = int(counts.get("steps/is_first", {}).get("count", 0))
            if steps < 1:
                raise ValueError("episode has no action/state steps")
            for key in camera_keys:
                if int(counts.get(key, {}).get("count", 0)) != steps:
                    raise ValueError(f"camera frame count differs from steps: {key}")
            for key, width in sensor_widths.items():
                if int(counts.get(key, {}).get("count", 0)) != steps * width:
                    raise ValueError(f"sensor shape differs from steps: {key}")
                episode_example_float_values(record, key)
            action_key = "steps/action_dict/cartesian_velocity"
            if int(counts.get(action_key, {}).get("count", 0)) != steps * 6:
                raise ValueError("Cartesian action target is not six values per step")
            normalized = sorted(
                {
                    " ".join("".join(ch.lower() if ch.isalnum() else " " for ch in text).split())
                    for text in episode_example_instructions(record)
                }
            )
            normalized = [value for value in normalized if value]
            instructions.update(normalized)
            rows.append(
                {
                    "episode_metadata/file_path": context["episode_metadata/file_path"],
                    "episode_metadata/recording_folderpath": context[
                        "episode_metadata/recording_folderpath"
                    ],
                    "normalized_instructions": normalized,
                    "step_count": steps,
                    "shard_index": index,
                }
            )
        if len(rows) != int(shard["episodes"]):
            raise ValueError(f"shard {index} episode count differs from pinned manifest")
        shard_counts[str(index)] = len(rows)
        episodes.extend(rows)
    if len(episodes) != int(manifest["expected_episode_count"]):
        raise ValueError("audited episode count differs from pinned total")
    return {
        "dataset_id": manifest["dataset_id"],
        "dataset_version": manifest["dataset_version"],
        "selection_manifest_sha256": manifest_sha,
        "episode_count": len(episodes),
        "unique_source_files": len(source_files),
        "unique_recording_groups": len(recording_groups),
        "unique_normalized_instructions": len(instructions),
        "normalized_instruction_repeats": sum(max(0, n - 1) for n in instructions.values()),
        "episodes_per_shard": shard_counts,
        "episodes": episodes,
        "sealed_audit_loaded": False,
        "source_partition": "official train; local development only",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--download-receipt", type=Path, required=True)
    parser.add_argument("--debug-audit", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    if output.is_relative_to(REPOSITORY_ROOT.resolve()):
        raise ValueError("experiment data must stay outside the repository")
    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    validate_manifest(manifest)
    manifest_sha = sha256(manifest_bytes)
    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes)
    if config["source"].get("selection_manifest_sha256") != manifest_sha:
        raise ValueError("experiment config belongs to another fresh-shard manifest")
    receipt_bytes = args.download_receipt.read_bytes()
    receipt = json.loads(receipt_bytes)
    if receipt.get("manifest_sha256") != manifest_sha:
        raise ValueError("download receipt belongs to another manifest")
    if receipt.get("episode_count") != manifest["expected_episode_count"]:
        raise ValueError("download receipt episode count mismatch")
    audit = _audit(args.data_dir.resolve(strict=True), manifest, manifest_sha)
    debug_audit = json.loads(args.debug_audit.read_text(encoding="utf-8"))
    split = build_group_split(
        audit,
        debug_audit,
        validation_groups=int(config["split"]["validation_groups"]),
        minimum_train_groups=int(config["split"]["minimum_train_groups"]),
        seed=int(config["split"]["seed"]),
    )

    split_ids = {
        name: set(split[name]["episode_ids"]) for name in ("train", "validation")
    }
    target_labels: dict[str, str] = {}
    audit_by_id = {
        stable_id(canonical_path(row["episode_metadata/file_path"])): row
        for row in audit["episodes"]
    }
    for shard in manifest["selected_shards"]:
        path = args.data_dir / manifest["prefix"] / Path(shard["object_name"]).name
        for record in iter_tfrecord(path):
            context = episode_example_context(record)
            episode_id = stable_id(canonical_path(context["episode_metadata/file_path"]))
            if episode_id not in audit_by_id:
                raise ValueError("target pass encountered a source absent from the audit")
            steps = int(audit_by_id[episode_id]["step_count"])
            observed_steps = int(steps * float(config["task"]["observed_fraction"]))
            observation_count = int(config["task"]["observation_count"])
            if observed_steps < observation_count:
                continue
            indices = uniform_indices(observed_steps, observation_count)
            target_index = indices[-1] + 1
            values = episode_example_float_values(
                record, "steps/action_dict/cartesian_velocity"
            )
            target_labels[episode_id] = label_from_velocity(values, steps, target_index)
    task = build_task_manifest(
        audit,
        split,
        target_labels,
        observation_count=int(config["task"]["observation_count"]),
        observed_fraction=float(config["task"]["observed_fraction"]),
    )
    train_rows = task["splits"]["train"]["rows"]
    validation_rows = task["splits"]["validation"]["rows"]
    if not train_rows or not validation_rows:
        raise ValueError("task split has no eligible train or validation examples")
    if not {row["id"] for row in train_rows} <= split_ids["train"]:
        raise ValueError("task train rows escaped the frozen train split")
    if not {row["id"] for row in validation_rows} <= split_ids["validation"]:
        raise ValueError("task validation rows escaped the frozen validation split")

    output.mkdir(parents=True, exist_ok=False)
    artifacts = {
        "audit.json": audit,
        "split.json": split,
        "task-manifest.json": task,
    }
    hashes = {}
    for filename, value in artifacts.items():
        raw = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
        (output / filename).write_bytes(raw)
        hashes[filename] = sha256(raw)
    for name, rows in (("train", train_rows), ("validation", validation_rows)):
        raw = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode()
        (output / f"{name}.jsonl").write_bytes(raw)
        hashes[f"{name}.jsonl"] = sha256(raw)
    run = {
        "status": "PREPARED_NOT_FEATURED_OR_FIT",
        "selection_manifest_sha256": manifest_sha,
        "experiment_config_sha256": sha256(config_bytes),
        "download_receipt_sha256": sha256(receipt_bytes),
        "debug_audit_sha256": sha256(args.debug_audit.read_bytes()),
        "artifact_sha256": hashes,
        "sealed_audit_loaded": False,
        "quality_fit_started": False,
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "task_target_counts": {
            split_name: task["splits"][split_name]["target_counts"]
            for split_name in ("train", "validation")
        },
    }
    (output / "run-metadata.json").write_text(json.dumps(run, indent=2) + "\n")
    print(json.dumps(run, indent=2))


if __name__ == "__main__":
    main()
