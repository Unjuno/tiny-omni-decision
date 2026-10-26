"""Measure trivial gripper-command persistence on the fixed DROID dev split."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.audit_droid100_groups import (
    episode_example_context,
    episode_example_float_values,
    iter_tfrecord,
)
from scripts.build_droid_group_split import _canonical_path
from scripts.download_droid_selected_shards import validate_manifest, verify_metadata

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_HASHES = {
    "manifest": "2bfb5626d203344bf2bd976f711c0ed5f7b4c1979932b7b48565329099207e64",
    "receipt": "2bfb6322e9c0c4649f875d8ad37fb3e990f786324fe2fdbd9cd6b2872af50667",
    "audit": "d60f850f88725a45f918a992685ab1174b38f086ea14029cac1ab665a504bc3b",
    "split": "d8d2e1ef353d2ea6388dd7e7574623e3b4d6518474afed14f042fc1838df7e43",
}
HISTORY_STEPS = 8
ANCHOR_STRIDE = 8
TARGET_HORIZON_STEPS = 1
POSITION_THRESHOLD = 0.5


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_pinned(path: Path, name: str) -> tuple[dict[str, Any], str]:
    data = path.read_bytes()
    digest = _sha256(data)
    if digest != EXPECTED_HASHES[name]:
        raise ValueError(f"{name} SHA-256 differs from the fixed development protocol")
    return json.loads(data), digest


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPOSITORY_ROOT
        / "manifests/pretrained_reuse/droid-selected-shards-seed17-20261026.json",
    )
    parser.add_argument("--receipt-json", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite persistence audit: {output}")
    if output.is_relative_to(REPOSITORY_ROOT.resolve()):
        raise ValueError("persistence audit output must be outside the repository")

    manifest_bytes = args.manifest.resolve(strict=True).read_bytes()
    manifest_sha = _sha256(manifest_bytes)
    if manifest_sha != EXPECTED_HASHES["manifest"]:
        raise ValueError("selected-shard manifest SHA-256 differs from the fixed protocol")
    manifest = json.loads(manifest_bytes)
    validate_manifest(manifest)
    receipt, receipt_sha = _load_pinned(args.receipt_json.resolve(strict=True), "receipt")
    audit, audit_sha = _load_pinned(args.audit_json.resolve(strict=True), "audit")
    split, split_sha = _load_pinned(args.split_json.resolve(strict=True), "split")
    if receipt.get("manifest_sha256") != manifest_sha:
        raise ValueError("download receipt references a different selection manifest")
    if audit.get("selection_manifest_sha256") != manifest_sha:
        raise ValueError("group audit references a different selection manifest")
    if split.get("selection_manifest_sha256") != manifest_sha:
        raise ValueError("development split references a different selection manifest")
    verify_metadata(args.data_dir.resolve(strict=True).parents[2], manifest)

    split_by_episode: dict[str, str] = {}
    for name in ("train", "validation"):
        for episode_id in split[name]["episode_ids"]:
            if episode_id in split_by_episode:
                raise ValueError("episode identity appears more than once in the split")
            split_by_episode[episode_id] = name
    audit_ids = {
        hashlib.sha256(
            _canonical_path(row["episode_metadata/file_path"]).encode("utf-8")
        ).hexdigest()
        for row in audit["episodes"]
    }
    if set(split_by_episode) != audit_ids:
        raise ValueError("split does not assign every audited episode exactly once")

    counts: dict[str, dict[str, Any]] = {
        name: {
            "episodes": 0,
            "samples": 0,
            "labels": Counter(),
            "state_match": 0,
            "prior_action_match": 0,
        }
        for name in ("train", "validation")
    }
    seen = set()
    for shard_row in manifest["selected_shards"]:
        shard_path = args.data_dir / Path(shard_row["object_name"]).name
        episode_rows = 0
        for record in iter_tfrecord(shard_path):
            context = episode_example_context(record)
            episode_id = hashlib.sha256(
                _canonical_path(context["episode_metadata/file_path"]).encode("utf-8")
            ).hexdigest()
            if episode_id not in split_by_episode:
                raise ValueError("source record is absent from the fixed development split")
            if episode_id in seen:
                raise ValueError("duplicate episode identity across selected shards")
            seen.add(episode_id)
            episode_rows += 1
            split_name = split_by_episode[episode_id]
            action = episode_example_float_values(record, "steps/action_dict/gripper_position")
            state = episode_example_float_values(record, "steps/observation/gripper_position")
            if len(action) != len(state):
                raise ValueError("gripper state and command series have different lengths")
            row = counts[split_name]
            row["episodes"] += 1
            for observation_index in range(
                HISTORY_STEPS - 1,
                len(action) - TARGET_HORIZON_STEPS,
                ANCHOR_STRIDE,
            ):
                target_index = observation_index + TARGET_HORIZON_STEPS
                target_value = action[target_index]
                if not 0 <= target_value <= 1:
                    raise ValueError("gripper position command lies outside the DROID [0,1] range")
                target = int(target_value >= POSITION_THRESHOLD)
                current_state = int(state[observation_index] >= POSITION_THRESHOLD)
                previous_command = int(action[observation_index] >= POSITION_THRESHOLD)
                row["samples"] += 1
                row["labels"][target] += 1
                row["state_match"] += target == current_state
                row["prior_action_match"] += target == previous_command
        if episode_rows != int(shard_row["episodes"]):
            raise ValueError("TFRecord episode count differs from pinned shard metadata")
    if seen != audit_ids:
        raise ValueError("downloaded records differ from the audited episode identity set")

    result = {
        "schema_version": 1,
        "status": "PERSISTENCE_BASELINE_ONLY",
        "claim_scope": "not a model result; candidate one-step gripper-choice feasibility",
        "selection_manifest_sha256": manifest_sha,
        "download_receipt_sha256": receipt_sha,
        "group_audit_sha256": audit_sha,
        "development_split_sha256": split_sha,
        "history_step_count": HISTORY_STEPS,
        "anchor_stride_steps": ANCHOR_STRIDE,
        "target_horizon_steps": TARGET_HORIZON_STEPS,
        "gripper_threshold": POSITION_THRESHOLD,
        "sealed_audit_loaded": False,
        "model_inference_or_training": False,
        "splits": {},
    }
    for name, row in counts.items():
        samples = row["samples"]
        label_counts = {str(key): value for key, value in sorted(row["labels"].items())}
        result["splits"][name] = {
            "episodes": row["episodes"],
            "windowed_samples": samples,
            "closed_target": row["labels"].get(1, 0),
            "open_target": row["labels"].get(0, 0),
            "majority_class_accuracy": max(row["labels"].values()) / samples,
            "current_state_persistence_accuracy": row["state_match"] / samples,
            "previous_command_persistence_accuracy": row["prior_action_match"] / samples,
            "label_counts": label_counts,
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
