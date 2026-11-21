"""Audit selected HABIT train/validation Parquet labels and causal video timing."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from bisect import bisect_left, bisect_right
from collections import Counter
from pathlib import Path
from typing import Any

from tiny_omni_decision.habit_decision import (
    habit_transition_points,
    map_habit_choice_target,
    normalize_habit_task_text,
)
from tiny_omni_decision.past_video_windows import past_window_indices

STATE_COLUMNS = (
    "robot0.observation.state.cartesian_position",
    "robot0.observation.state.joint_positions",
    "robot0.observation.state.gripper_position",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_selected_episode_metadata(path: Path, wanted: set[int]) -> dict[int, dict[str, Any]]:
    """Read full rows only for selected train/validation IDs; index-scan others."""
    found: dict[int, dict[str, Any]] = {}
    pattern = re.compile(r'^\s*\{\s*"episode_index"\s*:\s*(\d+)\s*,')
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            match = pattern.match(line)
            if not match:
                raise ValueError("pinned episode metadata changed its leading ID field")
            episode_id = int(match.group(1))
            if episode_id in wanted:
                row = json.loads(line)
                if row["episode_index"] != episode_id:
                    raise ValueError("episode metadata ID parse mismatch")
                found[episode_id] = row
    return found


def _read_selected_subtasks(path: Path, wanted: set[int]) -> dict[int, str]:
    """Read task text only for target IDs present in the selected payload."""
    found: dict[int, str] = {}
    pattern = re.compile(r'^\s*\{\s*"task_index"\s*:\s*(\d+)\s*,')
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            match = pattern.match(line)
            if not match:
                raise ValueError("pinned subtask metadata changed its leading ID field")
            target_id = int(match.group(1))
            if target_id in wanted:
                row = json.loads(line)
                if row["task_index"] != target_id:
                    raise ValueError("subtask ID parse mismatch")
                found[target_id] = row["task"]
    return found


def _assert_group_disjoint(rows: list[dict[str, Any]], split_by_id: dict[int, str],
                           task_details_dir: Path) -> int:
    """Check selected IDs using the same task/SID/instruction identity links."""
    sids = {str(row["sid"]) for row in rows}
    parent = {sid: sid for sid in sids}

    def find(sid: str) -> str:
        while parent[sid] != sid:
            parent[sid] = parent[parent[sid]]
            sid = parent[sid]
        return sid

    def union(values: set[str]) -> None:
        present = sorted(set(values) & sids)
        if present:
            root = find(present[0])
            for sid in present[1:]:
                other = find(sid)
                if other != root:
                    parent[other] = root

    for path in task_details_dir.glob("*.md"):
        union({part for part in path.stem.split("_") if part})
    by_task: dict[str, set[str]] = {}
    by_instruction: dict[str, set[str]] = {}
    for row in rows:
        sid = str(row["sid"])
        by_task.setdefault(str(row.get("task_index", "")), set()).add(sid)
        by_instruction.setdefault(
            normalize_habit_task_text(row["high_level_instruction"]), set()
        ).add(sid)
    for values in (*by_task.values(), *by_instruction.values()):
        union(values)
    split_by_root: dict[str, set[str]] = {}
    for row in rows:
        root = find(str(row["sid"]))
        split_by_root.setdefault(root, set()).add(split_by_id[row["episode_index"]])
    if any(len(splits) != 1 for splits in split_by_root.values()):
        raise ValueError("task/SID/instruction group crosses train and validation")
    return len(split_by_root)


def audit_pilot_payload(
    *,
    selection_path: Path,
    inventory_path: Path,
    download_receipt_path: Path,
    source_metadata_dir: Path,
    payload_dir: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Validate every selected example; never open sealed-audit payloads."""
    import av
    import pyarrow.parquet as pq

    selection_bytes = selection_path.read_bytes()
    selection = json.loads(selection_bytes)
    download = json.loads(download_receipt_path.read_text(encoding="utf-8"))
    inventory_bytes = inventory_path.read_bytes()
    inventory = json.loads(inventory_bytes)
    if output_path.exists():
        raise ValueError("output receipt must be new; preserve previous audit attempts")
    if selection["source_revision"] != "bae97d251907743f8fb6db7e70c61d361df4dc1e":
        raise ValueError("selection is not pinned to the screened HABIT revision")
    if download.get("sealed_audit_accessed") is not False:
        raise ValueError("download receipt indicates sealed-audit access")
    if (download.get("revision") != selection["source_revision"]
            or download.get("inventory_sha256") != hashlib.sha256(inventory_bytes).hexdigest()
            or inventory.get("selection_receipt_sha256")
            != hashlib.sha256(selection_bytes).hexdigest()):
        raise ValueError("selection, publisher inventory, and downloaded receipt disagree")
    for split, ids in (("train", selection["train_episode_ids"]),
                       ("validation", selection["validation_episode_ids"])):
        digest = hashlib.sha256("\n".join(map(str, sorted(ids))).encode()).hexdigest()
        if (selection["selected_episode_id_sha256"].get(split) != digest
                or inventory["selection_id_hashes"].get(split) != digest):
            raise ValueError(f"{split} episode IDs differ from their frozen hashes")
    if (inventory.get("payload_downloaded") is not False
            or inventory.get("sealed_audit_accessed") is not False):
        raise ValueError("publisher inventory must remain train/validation-only preflight")
    if download.get("cross_split_media_overlap") is not False:
        raise ValueError("exact video bytes overlap train and validation")
    selected = {
        "train": set(selection["train_episode_ids"]),
        "validation": set(selection["validation_episode_ids"]),
    }
    if selected["train"] & selected["validation"]:
        raise ValueError("selected episode IDs cross splits")
    expected_ids = selected["train"] | selected["validation"]
    metadata = _read_selected_episode_metadata(
        source_metadata_dir / "full_meta_episodes.jsonl", expected_ids
    )
    if set(metadata) != expected_ids:
        raise ValueError("selected train/validation episode metadata is incomplete")
    split_by_id = {
        episode_id: split for split, ids in selected.items() for episode_id in ids
    }
    source_rows = list(metadata.values())
    group_count = _assert_group_disjoint(
        source_rows, split_by_id, source_metadata_dir / "task_details"
    )
    items = download.get("files", [])
    inventory_items = {
        (item["split"], item["episode_index"], item["kind"]): item
        for item in inventory.get("items", [])
    }
    item_map = {(item["split"], item["episode_index"], item["kind"]): item for item in items}
    if len(item_map) != len(items):
        raise ValueError("download receipt contains duplicate file assignments")
    if len(items) != 2 * len(expected_ids):
        raise ValueError("download receipt is missing selected media or Parquet")
    if set(item_map) != set(inventory_items):
        raise ValueError("download receipt file assignments differ from inventory")
    for key, item in item_map.items():
        expected_item = inventory_items[key]
        if (item["path"] != expected_item["path"]
                or item["sha256"] != expected_item["sha256"]
                or item["bytes"] != expected_item["size"]):
            raise ValueError("downloaded payload differs from publisher inventory")

    selected_target_ids = set()
    for split, episode_id, kind in item_map:
        if kind == "parquet":
            path = payload_dir / split / "parquet" / f"episode_{episode_id:06d}.parquet"
            selected_target_ids.update(
                pq.read_table(path, columns=["low_level_task_index"])
                .column("low_level_task_index").to_pylist()
            )
    subtasks = _read_selected_subtasks(
        source_metadata_dir / "full_meta_subtasks.jsonl", selected_target_ids
    )
    if set(subtasks) != selected_target_ids:
        raise ValueError("selected Parquet labels are missing pinned task text")

    episode_results = []
    split_stats = {
        split: {"episodes": 0, "transition_examples": 0, "target_counts": Counter(),
                "option_counts": Counter(), "groups": set()}
        for split in selected
    }
    video_hash_splits: dict[str, set[str]] = {}
    for split, ids in selected.items():
        for episode_id in sorted(ids):
            row = metadata[episode_id]
            parquet_item = item_map[(split, episode_id, "parquet")]
            video_item = item_map[(split, episode_id, "video")]
            parquet_path = payload_dir / split / "parquet" / f"episode_{episode_id:06d}.parquet"
            video_path = payload_dir / split / "video" / f"episode_{episode_id:06d}.mp4"
            if (not parquet_path.is_file() or not video_path.is_file()
                    or _sha256(parquet_path) != parquet_item["sha256"]
                    or _sha256(video_path) != video_item["sha256"]):
                raise ValueError(f"payload hash mismatch for episode {episode_id}")
            video_hash_splits.setdefault(video_item["sha256"], set()).add(split)

            columns = ["timestamp", "low_level_task_index", *STATE_COLUMNS]
            table = pq.read_table(parquet_path, columns=columns)
            timestamps_s = table.column("timestamp").to_pylist()
            target_ids = table.column("low_level_task_index").to_pylist()
            if not timestamps_s or len(timestamps_s) != len(target_ids):
                raise ValueError(f"empty or misaligned Parquet rows in episode {episode_id}")
            if (any(not isinstance(t, (int, float)) or not math.isfinite(t) for t in timestamps_s)
                    or any(b <= a for a, b in zip(timestamps_s, timestamps_s[1:], strict=False))):
                raise ValueError(f"invalid Parquet timestamp series in episode {episode_id}")
            states = []
            for index in range(len(timestamps_s)):
                state = sum(
                    (table.column(name)[index].as_py() for name in STATE_COLUMNS), []
                )
                if len(state) != 14 or any(
                    not isinstance(value, (int, float)) or not math.isfinite(value)
                    for value in state
                ):
                    raise ValueError(f"invalid 14-D state in episode {episode_id}, row {index}")
                states.append(state)

            for target_id in sorted(set(target_ids)):
                map_habit_choice_target(row["low_level_tasks"], target_id, subtasks)
            option_result = map_habit_choice_target(
                row["low_level_tasks"], target_ids[0], subtasks
            )
            for target_id in target_ids:
                map_habit_choice_target(row["low_level_tasks"], target_id, subtasks)

            with av.open(str(video_path)) as container:
                if len(container.streams.video) != 1 or len(container.streams.audio):
                    raise ValueError(f"unexpected audio/video stream count in episode {episode_id}")
                stream = container.streams.video[0]
                video_times_s = [
                    float(frame.pts * stream.time_base)
                    for frame in container.decode(stream)
                    if frame.pts is not None
                ]
                width, height = stream.codec_context.width, stream.codec_context.height
                rate = float(stream.average_rate)
            if (width, height) != (640, 480) or not math.isclose(rate, 10.0, abs_tol=1e-6):
                raise ValueError(f"unexpected front-view video format in episode {episode_id}")
            if len(video_times_s) != len(timestamps_s):
                raise ValueError(f"video/Parquet frame counts differ in episode {episode_id}")
            if (any(not math.isfinite(t) for t in video_times_s)
                    or any(b <= a for a, b in zip(video_times_s, video_times_s[1:], strict=False))):
                raise ValueError(f"invalid video PTS series in episode {episode_id}")
            alignment_ms = max(
                min(
                    abs(t - video_times_s[candidate])
                    for candidate in (max(0, bisect_left(video_times_s, t) - 1),
                                      min(len(video_times_s) - 1, bisect_left(video_times_s, t)))
                )
                for t in timestamps_s
            ) * 1000
            if alignment_ms > 1.0:
                raise ValueError(f"video/Parquet clocks differ by {alignment_ms:.3f} ms")

            transitions = habit_transition_points(
                [t * 1000 for t in timestamps_s], target_ids, pre_transition_guard_ms=100.0
            )
            transition_rows = []
            for point in transitions:
                state_index = bisect_right(timestamps_s, point.context_end_ms / 1000) - 1
                if state_index < 0 or timestamps_s[state_index] * 1000 > point.context_end_ms:
                    raise ValueError(f"state after cutoff in episode {episode_id}")
                frame_indices = past_window_indices(
                    [t * 1000 for t in video_times_s],
                    start_ms=point.context_end_ms - 3200.0,
                    end_ms=point.context_end_ms,
                )
                latest_input_ms = max(video_times_s[i] * 1000 for i in frame_indices)
                if len(frame_indices) != 8 or latest_input_ms > point.context_end_ms:
                    raise ValueError(f"future or missing video frame in episode {episode_id}")
                transition_rows.append({
                    "target_subtask_id": point.target_subtask_id,
                    "context_end_ms": point.context_end_ms,
                    "onset_ms": point.onset_ms,
                    "state_row_index": state_index,
                    "state_age_ms": point.context_end_ms - timestamps_s[state_index] * 1000,
                    "video_frame_indices": frame_indices,
                    "target_option_index": map_habit_choice_target(
                        row["low_level_tasks"], point.target_subtask_id, subtasks
                    ).target_index,
                })

            stats = split_stats[split]
            stats["episodes"] += 1
            stats["transition_examples"] += len(transitions)
            stats["groups"].add(row["sid"])
            stats["option_counts"][len(option_result.options)] += 1
            stats["target_counts"].update(point.target_subtask_id for point in transitions)
            episode_results.append({
                "split": split,
                "episode_index": episode_id,
                "sid": row["sid"],
                "task_index": row["task_index"],
                "parquet_sha256": parquet_item["sha256"],
                "video_sha256": video_item["sha256"],
                "frame_count": len(video_times_s),
                "resolution": [width, height],
                "fps": rate,
                "video_parquet_max_time_error_ms": alignment_ms,
                "raw_option_count": len(row["low_level_tasks"]),
                "unique_option_count": len(option_result.options),
                "collapsed_duplicate_count": option_result.collapsed_duplicate_count,
                "unique_target_count": len(set(target_ids)),
                "state_width": 14,
                "transition_count": len(transitions),
                "transition_inputs": transition_rows,
                "action_column_read": False,
            })

    if any(len(splits) > 1 for splits in video_hash_splits.values()):
        raise ValueError("exact video content hash overlaps train and validation")
    summary = {
        split: {
            **{key: value for key, value in stats.items() if key != "target_counts"
               and key != "option_counts" and key != "groups"},
            "scenario_ids": len(stats["groups"]),
            "option_count_distribution": dict(sorted(stats["option_counts"].items())),
            "target_subtask_id_counts": dict(sorted(stats["target_counts"].items())),
        }
        for split, stats in split_stats.items()
    }
    report = {
        "dataset": "configinc/HABIT",
        "source_revision": selection["source_revision"],
        "selection_sha256": hashlib.sha256(selection_bytes).hexdigest(),
        "download_receipt_sha256": _sha256(download_receipt_path),
        "publisher_inventory_sha256": hashlib.sha256(inventory_bytes).hexdigest(),
        "payload_downloaded_bytes": download["bytes"],
        "sealed_audit_payload_accessed": False,
        "sealed_audit_metrics_generated": False,
        "episode_metadata_rows_parsed": len(metadata),
        "source_episode_metadata_scanned_for_split_id_only": True,
        "action_columns_read": False,
        "cross_split_media_hash_overlap": False,
        "task_group_count": group_count,
        "split_summary": summary,
        "episodes": episode_results,
    }
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--download-receipt", type=Path, required=True)
    parser.add_argument("--source-metadata-dir", type=Path, required=True)
    parser.add_argument("--payload-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_pilot_payload(
        selection_path=args.selection,
        inventory_path=args.inventory,
        download_receipt_path=args.download_receipt,
        source_metadata_dir=args.source_metadata_dir,
        payload_dir=args.payload_dir,
        output_path=args.output,
    )
    print(json.dumps(report["split_summary"], indent=2))


if __name__ == "__main__":
    main()
