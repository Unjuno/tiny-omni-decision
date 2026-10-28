"""Build a fresh, site-disjoint DROID task manifest from pinned source shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from scripts.audit_droid100_groups import (  # noqa: E402
    episode_example_bytes_values,
    episode_example_context,
    episode_example_float_values,
    iter_tfrecord,
)
from tiny_omni_decision.droid_next_action_readout import (  # noqa: E402
    ACTION_OPTIONS,
    next_cartesian_direction_label,
)
from tiny_omni_decision.droid_site_holdout import (  # noqa: E402
    droid_site_from_path,
    select_site_holdout,
    stable_droid_id,
    uniform_step_indices,
    validate_site_split,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--previous-task-manifest", type=Path, required=True)
    parser.add_argument("--split-protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _read_json(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value, raw


def main() -> None:
    args = _arguments()
    started = time.perf_counter()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite task output: {output}")
    if output.is_relative_to(REPOSITORY_ROOT.resolve()):
        raise ValueError("task output must remain outside the repository")

    selection, selection_raw = _read_json(args.selection_manifest)
    previous, previous_raw = _read_json(args.previous_task_manifest)
    protocol, protocol_raw = _read_json(args.split_protocol)
    if selection.get("sealed_audit_loaded") is not False:
        raise ValueError("source selection manifest indicates a sealed audit load")
    if protocol.get("labels_used_to_choose_split") is not False:
        raise ValueError("site assignment protocol must not use target labels")
    previous_splits = previous.get("splits", {})
    previous_rows = [
        row
        for split in ("train", "validation")
        for row in previous_splits.get(split, {}).get("rows", [])
    ]
    previous_ids = {row["id"] for row in previous_rows}
    previous_groups = {row["group_id"] for row in previous_rows}
    if len(previous_ids) != len(previous_rows) or len(previous_groups) != len(previous_rows):
        raise ValueError("previous train/validation source identities are not unique")

    data_root = args.dataset_root.resolve(strict=True) / selection["prefix"]
    shard_rows = selection.get("selected_shards", [])
    if not shard_rows or len({int(row["index"]) for row in shard_rows}) != len(shard_rows):
        raise ValueError("selected shard inventory is empty or contains duplicate indices")
    shards: list[tuple[dict[str, Any], Path]] = []
    for shard in sorted(shard_rows, key=lambda row: int(row["index"])):
        path = data_root / Path(shard["object_name"]).name
        if not path.is_file() or path.stat().st_size != int(shard["size"]):
            raise ValueError(f"source shard missing or wrong size: {path}")
        if sha256_file(path) != shard["sha256"]:
            raise ValueError(f"source shard SHA-256 mismatch: {path.name}")
        shards.append((shard, path))

    # First pass reads source identities and lengths only; it does not read target values.
    metadata: dict[str, dict[str, Any]] = {}
    group_ids: set[str] = set()
    for shard, path in shards:
        count = 0
        for record in iter_tfrecord(path):
            count += 1
            context = episode_example_context(record)
            source_path = context["episode_metadata/file_path"]
            group_path = context["episode_metadata/recording_folderpath"]
            identity = stable_droid_id(source_path)
            group_id = stable_droid_id(group_path)
            if identity in metadata or group_id in group_ids:
                raise ValueError("source episode or recording group repeats in selected shards")
            if identity in previous_ids or group_id in previous_groups:
                raise ValueError("new cohort overlaps the previous DROID train/validation set")
            actions = episode_example_float_values(
                record, "steps/action_dict/cartesian_velocity"
            )
            if len(actions) % 6:
                raise ValueError(f"Cartesian action array is malformed: {identity}")
            steps = len(actions) // 6
            site = droid_site_from_path(source_path)
            metadata[identity] = {
                "group_id": group_id,
                "site": site,
                "source_step_count": steps,
                "shard_index": int(shard["index"]),
            }
            group_ids.add(group_id)
        if count != int(shard["episodes"]):
            raise ValueError(f"source record count mismatch in {path.name}: {count}")

    eligible = {
        identity: row
        for identity, row in metadata.items()
        if row["source_step_count"] >= 32
        and row["source_step_count"] // 4 >= 8
        and row["source_step_count"] // 4 < row["source_step_count"]
    }
    site_counts = Counter(row["site"] for row in eligible.values())
    selected_split = select_site_holdout(site_counts)
    if selected_split["validation_sites"] != protocol.get("validation_sites"):
        raise ValueError("observed site counts disagree with the frozen held-out site list")
    if selected_split["validation_rows"] != int(
        protocol.get("validation_rows_by_step_count_only", -1)
    ):
        raise ValueError("eligible row count disagrees with the frozen split protocol")

    validation_sites = set(selected_split["validation_sites"])
    rows: list[dict[str, Any]] = []
    media_by_split: dict[str, set[str]] = {"train": set(), "validation": set()}
    target_counts: dict[str, Counter[str]] = {
        "train": Counter(),
        "validation": Counter(),
    }
    sites_by_split: dict[str, Counter[str]] = {
        "train": Counter(),
        "validation": Counter(),
    }
    view_keys = [
        "steps/observation/exterior_image_1_left",
        "steps/observation/exterior_image_2_left",
        "steps/observation/wrist_image_left",
    ]
    sensor_fields = (
        ("steps/observation/joint_position", 7),
        ("steps/observation/cartesian_position", 6),
        ("steps/observation/gripper_position", 1),
    )
    seen: set[str] = set()
    for _shard, path in shards:
        for record in iter_tfrecord(path):
            context = episode_example_context(record)
            identity = stable_droid_id(context["episode_metadata/file_path"])
            metadata_row = metadata.get(identity)
            if metadata_row is None or identity not in eligible:
                continue
            if identity in seen:
                raise ValueError(f"duplicate task episode: {identity}")
            seen.add(identity)
            steps = int(metadata_row["source_step_count"])
            observed = steps // 4
            indices = uniform_step_indices(observed, 8)
            target_index = indices[-1] + 1
            if target_index != observed or target_index >= steps:
                raise ValueError("task target is not immediately after the observed window")
            actions = np.asarray(
                episode_example_float_values(
                    record, "steps/action_dict/cartesian_velocity"
                ),
                dtype=np.float64,
            ).reshape(steps, 6)
            if not np.isfinite(actions).all():
                raise ValueError(f"non-finite commanded velocity: {identity}")
            label = ACTION_OPTIONS[next_cartesian_direction_label(actions[target_index])]
            for key, width in sensor_fields:
                values = episode_example_float_values(record, key)
                if len(values) != steps * width or not np.isfinite(values).all():
                    raise ValueError(f"sensor sequence shape/value mismatch: {identity} {key}")
            frame_hashes: list[str] = []
            for key in view_keys:
                frames = episode_example_bytes_values(record, key)
                if len(frames) != steps:
                    raise ValueError(f"video sequence length mismatch: {identity} {key}")
                frame_hashes.extend(
                    hashlib.sha256(frames[index]).hexdigest() for index in indices
                )
            split = "validation" if metadata_row["site"] in validation_sites else "train"
            row = {
                "id": identity,
                "group_id": metadata_row["group_id"],
                "split": split,
                "site": metadata_row["site"],
                "shard_index": metadata_row["shard_index"],
                "source_step_count": steps,
                "observed_step_count": observed,
                "input_indices": indices,
                "target_index": target_index,
                "target_option_id": label,
                "options": list(ACTION_OPTIONS),
                "input_frame_sha256": frame_hashes,
            }
            rows.append(row)
            media_by_split[split].update(frame_hashes)
            target_counts[split][label] += 1
            sites_by_split[split][metadata_row["site"]] += 1
    if seen != set(eligible):
        raise ValueError(f"task pass missed {len(set(eligible) - seen)} eligible episodes")
    rows.sort(key=lambda row: (row["split"], row["id"]))
    validate_site_split(rows)

    task = {
        "schema_version": 1,
        "task_id": "droid-r2-expanded-site-heldout-next-action-v1",
        "claim_scope": (
            "Development-only DROID one-step Cartesian command-direction prediction; "
            "not a general omni benchmark."
        ),
        "source_partition": (
            "Official Google Research DROID r2d2_faceblur 1.0.0 train shards "
            "00000..00019 only."
        ),
        "source_commit": "eba818067c6a9ef9089288205c26f6a8254ca6e5",
        "options": list(ACTION_OPTIONS),
        "target_definition": (
            "Dominant signed xyz axis of commanded Cartesian velocity at the step "
            "immediately after the last observed state; stop when max abs(xyz) <= 1e-4."
        ),
        "input_contract": {
            "modalities": ["video", "sensor"],
            "video_views": view_keys,
            "sensor_fields": [
                "observation/joint_position",
                "observation/cartesian_position",
                "observation/gripper_position",
            ],
            "observation_count": 8,
            "observed_fraction": 0.25,
            "timebase": "uniform source step indices; no per-step wall-clock timestamps asserted",
            "future_observations_in_input": False,
        },
        "split_protocol_sha256": hashlib.sha256(protocol_raw).hexdigest(),
        "sealed_audit_loaded": False,
        "splits": {},
    }
    for split in ("train", "validation"):
        split_rows = [row for row in rows if row["split"] == split]
        task["splits"][split] = {
            "episode_count": len(split_rows),
            "group_count": len({row["group_id"] for row in split_rows}),
            "site_count": len(sites_by_split[split]),
            "site_counts": dict(sorted(sites_by_split[split].items())),
            "target_counts": {
                option: target_counts[split][option] for option in ACTION_OPTIONS
            },
            "rows": split_rows,
        }
    output.mkdir(parents=True, exist_ok=False)
    task_text = json.dumps(task, indent=2, sort_keys=True) + "\n"
    task_raw = task_text.replace("\n", "\r\n").encode("utf-8")
    task_path = output / "task-manifest.json"
    task_path.write_bytes(task_raw)
    media_overlap = media_by_split["train"] & media_by_split["validation"]
    summary = {
        "status": "PASS_SOURCE_ID_AND_GROUP_AUDIT",
        "selection_manifest_sha256": hashlib.sha256(selection_raw).hexdigest(),
        "previous_task_manifest_sha256": hashlib.sha256(previous_raw).hexdigest(),
        "split_protocol_sha256": hashlib.sha256(protocol_raw).hexdigest(),
        "task_manifest_sha256": hashlib.sha256(task_raw).hexdigest(),
        "source_records_total": len(metadata),
        "eligible_rows": len(rows),
        "excluded_short_rows": len(metadata) - len(rows),
        "train_rows": task["splits"]["train"]["episode_count"],
        "validation_rows": task["splits"]["validation"]["episode_count"],
        "train_sites": task["splits"]["train"]["site_counts"],
        "validation_sites": task["splits"]["validation"]["site_counts"],
        "train_validation_id_or_group_overlap": False,
        "train_validation_exact_jpeg_hash_overlap": len(media_overlap),
        "labels_used_to_choose_split": False,
        "sealed_audit_loaded": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    if media_overlap:
        raise ValueError(f"exact JPEG content overlaps across splits: {len(media_overlap)}")
    (output / "task-audit.json").write_bytes(
        (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
