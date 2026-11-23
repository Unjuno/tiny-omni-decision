"""Audit one four-episode action-option group in each frozen partition.

Requires optional local packages PyArrow and PyAV. It reads only the explicitly
selected publisher-TRAIN files and produces no model score or feature cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import av
import pyarrow.parquet as pq

from tiny_omni_decision.robopro_actions import numeric_option_separation

SAMPLE_EPISODES = {
    "train": [7801, 7802, 7805, 7807],
    "development": [5001, 5005, 5006, 5008],
}
QUERY_PROGRESS = (0.25, 0.5, 0.75)
ACTION_DIMENSION = 14
FPS = 50
TIME_TOLERANCE_SECONDS = 0.001


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def audit_episode(
    episode_id: int,
    split: str,
    prompt: str,
    expected_length: int,
    task_id_to_text: dict[int, str],
    data_root: Path,
) -> dict[str, Any]:
    parquet_path = data_root / split / "data" / f"episode_{episode_id:06d}.parquet"
    video_path = data_root / split / "video" / f"episode_{episode_id:06d}_cam_high.mp4"
    if not parquet_path.is_file() or not video_path.is_file():
        raise FileNotFoundError(f"missing selected episode files for {split}:{episode_id}")

    table = pq.read_table(parquet_path)
    rows = table.to_pylist()
    if len(rows) != expected_length:
        raise ValueError(f"Parquet row count does not match metadata for episode {episode_id}")
    frame_indices = [row["frame_index"] for row in rows]
    if frame_indices != list(range(expected_length)):
        raise ValueError(f"frame indices are not contiguous for episode {episode_id}")
    if {row["episode_index"] for row in rows} != {episode_id}:
        raise ValueError(f"episode id changed inside Parquet for episode {episode_id}")

    state = [row["observation.state"] for row in rows]
    actions = [row["action"] for row in rows]
    if any(len(vector) != ACTION_DIMENSION for vector in state + actions):
        raise ValueError(f"non-14D state/action vector in episode {episode_id}")
    if any(not math.isfinite(value) for vector in state + actions for value in vector):
        raise ValueError(f"non-finite state/action value in episode {episode_id}")

    timestamps = [float(row["timestamp"]) for row in rows]
    timestamp_deltas = [
        right - left for left, right in zip(timestamps[:-1], timestamps[1:], strict=True)
    ]
    if not timestamp_deltas or max(abs(delta - 1 / FPS) for delta in timestamp_deltas) > 1e-4:
        raise ValueError(f"Parquet timestamps are not at {FPS} Hz in episode {episode_id}")
    task_ids = {int(row["task_index"]) for row in rows}
    if len(task_ids) != 1 or task_id_to_text[next(iter(task_ids))] != prompt:
        raise ValueError(f"task_index does not match prompt for episode {episode_id}")

    with av.open(str(video_path)) as container:
        video_stream = container.streams.video[0]
        stream_rate = float(video_stream.average_rate)
        video_times = [
            float(frame.pts * video_stream.time_base)
            for frame in container.decode(video=video_stream.index)
        ]
    max_pts_delta = (
        max(abs(timestamp - pts) for timestamp, pts in zip(timestamps, video_times, strict=True))
        if len(video_times) == len(timestamps)
        else None
    )
    if len(video_times) != expected_length or abs(stream_rate - FPS) > 1e-6:
        raise ValueError(f"decoded video frame count/rate mismatch for episode {episode_id}")
    if max_pts_delta is None or max_pts_delta > TIME_TOLERANCE_SECONDS:
        raise ValueError(f"video PTS does not align with Parquet for episode {episode_id}")

    return {
        "episode_id": episode_id,
        "split": split,
        "prompt": prompt,
        "rows": len(rows),
        "task_index": next(iter(task_ids)),
        "state_dimension": ACTION_DIMENSION,
        "action_dimension": ACTION_DIMENSION,
        "timestamp_first_seconds": timestamps[0],
        "timestamp_last_seconds": timestamps[-1],
        "median_timestamp_delta_seconds": sorted(timestamp_deltas)[len(timestamp_deltas) // 2],
        "video_frames": len(video_times),
        "video_fps": stream_rate,
        "max_abs_pts_timestamp_delta_seconds": max_pts_delta,
        "parquet": {
            "bytes": parquet_path.stat().st_size,
            "sha256": sha256_file(parquet_path),
        },
        "video": {
            "bytes": video_path.stat().st_size,
            "sha256": sha256_file(video_path),
        },
        "_actions": actions,
    }


def build_audit(metadata_dir: Path, split_manifest_path: Path, data_root: Path) -> dict[str, Any]:
    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    for name, expected in split_manifest["source_files"].items():
        path = metadata_dir / name
        if sha256_file(path) != expected["sha256"]:
            raise ValueError(f"source metadata hash mismatch: {name}")
    tasks = _read_jsonl(metadata_dir / "tasks.jsonl")
    episodes = _read_jsonl(metadata_dir / "episodes.jsonl")
    task_id_to_text = {int(row["task_index"]): row["task"] for row in tasks}
    episode_by_id = {int(row["episode_index"]): row for row in episodes}

    partitions: dict[str, Any] = {}
    for split, sample_ids in SAMPLE_EPISODES.items():
        partition_ids = set(split_manifest[split]["episode_ids"])
        if not set(sample_ids) <= partition_ids:
            raise ValueError(f"sample episodes are outside the frozen {split} split")
        prompts = {episode_by_id[episode_id]["tasks"][0] for episode_id in sample_ids}
        if len(prompts) != 1:
            raise ValueError(f"selected {split} episodes do not share one exact prompt")
        prompt = next(iter(prompts))
        prompt_groups = set(split_manifest[split]["prompt_groups"])
        if prompt not in prompt_groups:
            raise ValueError(f"selected {split} prompt is absent from the split manifest")

        audited: list[dict[str, Any]] = []
        for episode_id in sample_ids:
            metadata = episode_by_id[episode_id]
            audited.append(
                audit_episode(
                    episode_id,
                    split,
                    prompt,
                    int(metadata["length"]),
                    task_id_to_text,
                    data_root,
                )
            )

        phases: list[dict[str, Any]] = []
        for progress in QUERY_PROGRESS:
            option_vectors = []
            option_rows = []
            for item in audited:
                actions = item["_actions"]
                row_index = round((len(actions) - 1) * progress)
                option_vectors.append(actions[row_index])
                option_rows.append({"episode_id": item["episode_id"], "row_index": row_index})
            separation = numeric_option_separation(option_vectors)
            phases.append(
                {
                    "relative_progress": progress,
                    "query_rows": option_rows,
                    "action_option_separation": separation,
                    "all_four_options_distinct_within_tolerance": not separation[
                        "within_tolerance_pairs"
                    ],
                }
            )
        for item in audited:
            item.pop("_actions", None)
        partitions[split] = {
            "prompt": prompt,
            "episode_ids": sample_ids,
            "episodes": audited,
            "action_option_phases": phases,
            "candidate_group_integrity_pass": True,
            "all_phases_have_four_distinct_options": all(
                phase["all_four_options_distinct_within_tolerance"] for phase in phases
            ),
        }

    return {
        "schema_version": "robopro-action-option-smoke-v1",
        "dataset": split_manifest["dataset"],
        "revision": split_manifest["revision"],
        "scope": (
            "8 selected publisher-TRAIN episodes; metadata, action-vector, "
            "and video-clock integrity only"
        ),
        "model_or_features_loaded": False,
        "official_test_or_sealed_audit_accessed": False,
        "gpu_used": False,
        "source_metadata_hashes": split_manifest["source_files"],
        "split_manifest_sha256": sha256_file(split_manifest_path),
        "time_alignment_tolerance_seconds": TIME_TOLERANCE_SECONDS,
        "query_relative_progress": list(QUERY_PROGRESS),
        "partitions": partitions,
        "limitations": [
            "Action-option uniqueness is a numeric integrity check, "
            "not a validity or quality result.",
            "Logged commands are behavior targets; alternatives may also be valid.",
            "This eight-episode smoke does not establish dataset-wide media "
            "validity or synchronization.",
            "No scene-held-out split is available in the inspected source metadata.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = build_audit(args.metadata_dir, args.split_manifest, args.data_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                split: {
                    "episodes": len(partition["episodes"]),
                    "prompt": partition["prompt"],
                    "all_phases_have_four_distinct_options": partition[
                        "all_phases_have_four_distinct_options"
                    ],
                    "max_pts_delta": max(
                        episode["max_abs_pts_timestamp_delta_seconds"]
                        for episode in partition["episodes"]
                    ),
                }
                for split, partition in report["partitions"].items()
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
