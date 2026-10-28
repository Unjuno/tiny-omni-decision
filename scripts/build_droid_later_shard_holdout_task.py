"""Build a fixed, episode-disjoint DROID transfer evaluation manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

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
    stable_droid_id,
    uniform_step_indices,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> tuple[Any, bytes]:
    raw = path.read_bytes()
    return json.loads(raw.decode("utf-8-sig")), raw


def parse_input(value: str) -> tuple[int, Path]:
    try:
        index_text, path_text = value.split("=", 1)
        index = int(index_text)
    except (ValueError, TypeError) as error:
        raise ValueError("input must be INDEX=PATH") from error
    if index < 0 or not path_text:
        raise ValueError("input must contain a non-negative index and a path")
    return index, Path(path_text)


def build_manifest(
    *,
    inputs: list[tuple[int, Path]],
    source_audit_path: Path,
    selection_path: Path,
    previous_task_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    output = output_path.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite task manifest: {output}")
    if output.is_relative_to(ROOT.resolve()):
        raise ValueError("task manifest must be written outside the repository")

    audit, audit_raw = read_json(source_audit_path)
    selection, selection_raw = read_json(selection_path)
    previous, previous_raw = read_json(previous_task_path)
    if (
        audit.get("status") != "PASS_SOURCE_METADATA_AND_IDENTITY"
        or audit.get("labels_read") is not False
        or audit.get("media_decoded") is not False
        or audit.get("sealed_audit_loaded") is not False
        or audit.get("new_sites_in_selected_sample") != []
    ):
        raise ValueError("source-only audit does not satisfy the frozen sample gate")
    if selection.get("indices") != audit.get("selection_indices"):
        raise ValueError("sample selection and source audit differ")
    expected_indices = [int(value) for value in audit["selection_indices"]]
    if sorted(index for index, _ in inputs) != expected_indices:
        raise ValueError("exactly the frozen later-shard sample is required")

    previous_rows = [
        row
        for split_name in ("train", "validation")
        for row in previous["splits"][split_name]["rows"]
    ]
    previous_ids = {str(row["id"]) for row in previous_rows}
    previous_groups = {str(row["group_id"]) for row in previous_rows}
    previous_media = {
        str(value)
        for row in previous_rows
        for value in row.get("input_frame_sha256", [])
    }
    if len(previous_ids) != len(previous_rows) or len(previous_groups) != len(previous_rows):
        raise ValueError("previous train/validation identities are not unique")

    expected_shards = {int(row["index"]): row for row in audit["shards"]}
    selected_objects = {int(row["index"]): row for row in selection["objects"]}
    id_to_shard: dict[str, int] = {}
    records_by_id: dict[str, dict[str, Any]] = {}
    for index, path in sorted(inputs):
        descriptor = expected_shards.get(index)
        source_object = selected_objects.get(index)
        if descriptor is None or source_object is None:
            raise ValueError(f"sample audit lacks shard {index}")
        if (
            int(source_object["size"]) != int(descriptor["bytes"])
            or f"-{index:05d}-of-" not in source_object["object"]
        ):
            raise ValueError(f"pinned selection metadata differs for shard {index}")
        if not path.is_file() or path.stat().st_size != int(descriptor["bytes"]):
            raise ValueError(f"source shard missing or wrong size: {path}")
        if sha256_file(path) != descriptor["sha256"]:
            raise ValueError(f"source shard SHA-256 mismatch: {path.name}")
        record_count = 0
        for record in iter_tfrecord(path):
            record_count += 1
            context = episode_example_context(record)
            identity = stable_droid_id(context["episode_metadata/file_path"])
            group_id = stable_droid_id(context["episode_metadata/recording_folderpath"])
            if identity in previous_ids or group_id in previous_groups:
                raise ValueError("new sample overlaps a previous episode or recording group")
            if identity in records_by_id:
                raise ValueError("duplicate episode identity across sample shards")
            records_by_id[identity] = {
                "record": record,
                "context": context,
                "group_id": group_id,
            }
            id_to_shard[identity] = index
        if record_count != int(descriptor["records"]):
            raise ValueError(f"source shard record count mismatch: {index}")

    rows: list[dict[str, Any]] = []
    all_media: set[str] = set()
    skipped_short = 0
    site_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    camera_fields = (
        "steps/observation/exterior_image_1_left",
        "steps/observation/exterior_image_2_left",
        "steps/observation/wrist_image_left",
    )
    sensor_fields = (
        ("steps/observation/joint_position", 7),
        ("steps/observation/cartesian_position", 6),
        ("steps/observation/gripper_position", 1),
    )
    for identity in sorted(records_by_id):
        item = records_by_id[identity]
        record = item["record"]
        context = item["context"]
        action_values = episode_example_float_values(
            record, "steps/action_dict/cartesian_velocity"
        )
        if len(action_values) % 6:
            raise ValueError(f"malformed Cartesian action sequence: {identity}")
        steps = len(action_values) // 6
        observed = steps // 4
        if steps < 32 or observed < 8 or observed >= steps:
            skipped_short += 1
            continue
        indices = uniform_step_indices(observed, 8)
        target_index = indices[-1] + 1
        actions = np.asarray(action_values, dtype=np.float64).reshape(steps, 6)
        if not np.isfinite(actions).all():
            raise ValueError(f"non-finite action sequence: {identity}")
        option = ACTION_OPTIONS[next_cartesian_direction_label(actions[target_index])]
        frame_hashes: list[str] = []
        for key in camera_fields:
            frames = episode_example_bytes_values(record, key)
            if len(frames) != steps:
                raise ValueError(f"frame count mismatch: {identity} {key}")
            frame_hashes.extend(hashlib.sha256(frames[i]).hexdigest() for i in indices)
        if previous_media.intersection(frame_hashes):
            raise ValueError(f"fresh sample has frame content overlap with prior task: {identity}")
        if all_media.intersection(frame_hashes):
            raise ValueError(f"duplicate frame content across fresh sample episodes: {identity}")
        all_media.update(frame_hashes)
        for key, width in sensor_fields:
            values = episode_example_float_values(record, key)
            if len(values) != steps * width or not np.isfinite(values).all():
                raise ValueError(f"invalid sensor sequence: {identity} {key}")
        source_path = context["episode_metadata/file_path"]
        row = {
            "id": identity,
            "group_id": item["group_id"],
            "split": "validation",
            "site": droid_site_from_path(source_path),
            "shard_index": id_to_shard[identity],
            "source_step_count": steps,
            "observed_step_count": observed,
            "input_indices": indices,
            "target_index": target_index,
            "target_option_id": option,
            "options": list(ACTION_OPTIONS),
            "input_frame_sha256": frame_hashes,
        }
        rows.append(row)
        site_counts[row["site"]] += 1
        label_counts[option] += 1

    expected_eligible = int(audit["eligible_count_in_selected_shards"])
    if len(rows) != expected_eligible:
        raise ValueError(f"eligible rows {len(rows)} != source audit {expected_eligible}")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("duplicate task IDs")
    if len({row["group_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate recording groups")
    if len(all_media) != 24 * len(rows):
        raise ValueError("fresh task contains duplicate sampled media hashes")

    task = {
        "schema_version": 1,
        "task_id": "droid-later-shard-frozen-readout-transfer-v1",
        "claim_scope": (
            "One-time same-site, recording-group-disjoint DROID next-action transfer "
            "evaluation of frozen readouts; not blind or general Omni quality."
        ),
        "source_partition": (
            "Google Research DROID 1.0.0 official train shards "
            "00466, 00759, 01632."
        ),
        "source_license": "CC-BY-4.0",
        "source_shards": [
            {
                "index": index,
                "object": selected_objects[index]["object"],
                "size": int(expected_shards[index]["bytes"]),
                "sha256": expected_shards[index]["sha256"],
                "episodes": int(expected_shards[index]["records"]),
            }
            for index in expected_indices
        ],
        "sample_selection_sha256": audit["selection_sha256"],
        "source_audit_sha256": hashlib.sha256(audit_raw).hexdigest(),
        "selection_file_sha256": hashlib.sha256(selection_raw).hexdigest(),
        "previous_task_manifest_sha256": hashlib.sha256(previous_raw).hexdigest(),
        "labels_opened_after_model_freeze": True,
        "model_training": False,
        "sealed_audit_loaded": False,
        "options": list(ACTION_OPTIONS),
        "target_definition": (
            "Dominant signed xyz axis of commanded Cartesian velocity immediately "
            "after the eighth sampled observation; stop when max abs(xyz) <= 1e-4."
        ),
        "input_contract": {
            "modalities": ["video", "sensor"],
            "video_views": list(camera_fields),
            "sensor_fields": [key.removeprefix("steps/") for key, _ in sensor_fields],
            "observation_count": 8,
            "observed_fraction": 0.25,
            "timebase": "uniform source step indices; no per-step wall-clock timestamps asserted",
            "future_observations_in_input": False,
        },
        "split_protocol": {
            "unit": "episode/recording group",
            "train_rows": 0,
            "evaluation_rows": len(rows),
            "all_rows_are_unseen_official_train_groups": True,
            "source_group_overlap_with_previous_task": 0,
            "sampled_frame_overlap_with_previous_task": 0,
            "new_collection_sites": sorted(
                set(site_counts)
                - set(previous["splits"]["train"].get("site_counts", {}))
                - set(previous["splits"]["validation"].get("site_counts", {}))
            ),
            "site_counts": dict(sorted(site_counts.items())),
        },
        "skipped_short_episodes": skipped_short,
        "label_counts": dict(sorted(label_counts.items())),
        "splits": {
            "train": {"rows": [], "example_count": 0},
            "validation": {
                "rows": rows,
                "example_count": len(rows),
                "site_counts": dict(sorted(site_counts.items())),
            },
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(task, indent=2) + "\n", encoding="utf-8")
    return task


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", action="append", required=True, metavar="INDEX=PATH")
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--previous-task", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    task = build_manifest(
        inputs=[parse_input(value) for value in args.inputs],
        source_audit_path=args.source_audit,
        selection_path=args.selection,
        previous_task_path=args.previous_task,
        output_path=args.output,
    )
    print(json.dumps({
        "status": "PASS_FROZEN_TRANSFER_TASK",
        "task_id": task["task_id"],
        "evaluation_rows": task["splits"]["validation"]["example_count"],
        "skipped_short_episodes": task["skipped_short_episodes"],
        "site_counts": task["split_protocol"]["site_counts"],
        "label_counts": task["label_counts"],
        "new_collection_sites": task["split_protocol"]["new_collection_sites"],
        "sealed_audit_loaded": task["sealed_audit_loaded"],
    }, indent=2))


if __name__ == "__main__":
    main()
