"""Verify a frozen RoboPro probe sample before extracting features or fitting."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import av
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from tiny_omni_decision.robopro_actions import numeric_option_separation

FPS = 50
QUERY_PROGRESS = (0.25, 0.5, 0.75)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def audit(
    sample_path: Path,
    split_manifest_path: Path,
    metadata_dir: Path,
    data_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite audit report: {output_path}")
    sample_bytes = sample_path.read_bytes()
    sample = json.loads(sample_bytes)
    split_bytes = split_manifest_path.read_bytes()
    split_manifest = json.loads(split_bytes)
    episodes_path = metadata_dir / "episodes.jsonl"
    tasks_path = metadata_dir / "tasks.jsonl"
    if sha256_file(episodes_path) != sample["source_metadata_sha256"]:
        raise ValueError("frozen source metadata hash mismatch")
    if hashlib.sha256(split_bytes).hexdigest() != sample["split_manifest_sha256"]:
        raise ValueError("frozen split manifest hash mismatch")
    if split_manifest["revision"] != sample["revision"]:
        raise ValueError("source revision mismatch")
    receipt_path = data_root / "download-receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt["sample_manifest_sha256"] != hashlib.sha256(sample_bytes).hexdigest():
        raise ValueError("download receipt does not bind to the sample manifest")
    receipt_by_key = {
        (row["split"], int(row["episode_id"]), row["kind"]): row for row in receipt["files"]
    }
    tasks = {int(row["task_index"]): str(row["task"]) for row in _read_jsonl(tasks_path)}
    metadata = {int(row["episode_index"]): row for row in _read_jsonl(episodes_path)}
    all_split_ids = {
        split: set(map(int, split_manifest[split]["episode_ids"]))
        for split in ("train", "development")
    }
    selected_prompt_hashes = {}
    group_reports = []
    episode_reports = []
    normalized_frame_splits: dict[str, set[str]] = defaultdict(set)
    file_hash_splits: dict[str, set[str]] = defaultdict(set)
    for split in ("train", "development"):
        prompt_hashes = {group["prompt_sha256"] for group in sample[split]}
        if len(prompt_hashes) != len(sample[split]):
            raise ValueError(f"duplicate exact-prompt group in {split}")
        selected_prompt_hashes[split] = prompt_hashes
        for group in sample[split]:
            prompt = str(group["prompt"])
            if hashlib.sha256(prompt.encode("utf-8")).hexdigest() != group["prompt_sha256"]:
                raise ValueError("normalized prompt hash mismatch")
            records = group["episodes"]
            if len(records) != 4:
                raise ValueError("each fixed action-choice group must have four episodes")
            group_episodes = []
            for record in records:
                episode_id = int(record["episode_id"])
                source = metadata[episode_id]
                if episode_id not in all_split_ids[split]:
                    raise ValueError(f"episode {episode_id} is not in frozen {split} partition")
                if source["tasks"] != [prompt] or int(source["length"]) != int(record["length"]):
                    raise ValueError(f"sample metadata mismatch for episode {episode_id}")
                parquet_path = data_root / split / "data" / f"episode_{episode_id:06d}.parquet"
                video_path = data_root / split / "video" / f"episode_{episode_id:06d}_cam_high.mp4"
                entry_data = receipt_by_key[(split, episode_id, "parquet")]
                entry_video = receipt_by_key[(split, episode_id, "mp4")]
                if sha256_file(parquet_path) != entry_data["sha256"]:
                    raise ValueError(f"Parquet receipt hash mismatch: {episode_id}")
                if sha256_file(video_path) != entry_video["sha256"]:
                    raise ValueError(f"video receipt hash mismatch: {episode_id}")
                table = pq.read_table(parquet_path)
                rows = table.to_pylist()
                count = int(record["length"])
                if len(rows) != count:
                    raise ValueError(f"row count mismatch: {episode_id}")
                if [int(row["frame_index"]) for row in rows] != list(range(count)):
                    raise ValueError(f"frame indices are not contiguous: {episode_id}")
                if {int(row["episode_index"]) for row in rows} != {episode_id}:
                    raise ValueError(f"episode ID changes within Parquet: {episode_id}")
                if len({int(row["task_index"]) for row in rows}) != 1:
                    raise ValueError(f"task ID changes within Parquet: {episode_id}")
                task_id = int(rows[0]["task_index"])
                if tasks[task_id] != prompt:
                    raise ValueError(f"task index does not resolve to prompt: {episode_id}")
                states = np.asarray([row["observation.state"] for row in rows], dtype=np.float32)
                actions = np.asarray([row["action"] for row in rows], dtype=np.float32)
                stamps = np.asarray([row["timestamp"] for row in rows], dtype=np.float64)
                if states.shape != (count, 14) or actions.shape != (count, 14):
                    raise ValueError(f"non-14D state/action shape: {episode_id}")
                if not np.isfinite(states).all() or not np.isfinite(actions).all():
                    raise ValueError(f"non-finite state/action: {episode_id}")
                if np.max(np.abs(np.diff(stamps) - 1 / FPS)) > 1e-4:
                    raise ValueError(f"Parquet timestamp cadence differs from {FPS} Hz")
                with av.open(str(video_path)) as container:
                    stream = container.streams.video[0]
                    video_fps = float(stream.average_rate)
                    pts = []
                    content_hashes = []
                    for frame in container.decode(stream):
                        if frame.pts is None:
                            raise ValueError(f"video frame is missing PTS: {episode_id}")
                        pts.append(float(frame.pts * frame.time_base))
                        rgb = frame.to_ndarray(format="rgb24")
                        thumb = Image.fromarray(rgb).resize((32, 32), Image.Resampling.BILINEAR)
                        content_hashes.append(hashlib.sha256(thumb.tobytes()).hexdigest())
                if len(pts) != count or abs(video_fps - FPS) > 1e-6:
                    raise ValueError(f"video frame count/rate mismatch: {episode_id}")
                max_clock_delta = float(np.max(np.abs(stamps - np.asarray(pts))))
                if max_clock_delta > 0.001:
                    raise ValueError(f"video/Parquet clock mismatch: {episode_id}")
                for content_hash in set(content_hashes):
                    normalized_frame_splits[content_hash].add(split)
                file_hash_splits[entry_video["sha256"]].add(split)
                episode = {
                    "episode_id": episode_id,
                    "actions": actions,
                    "length": count,
                    "max_video_parquet_time_delta_seconds": max_clock_delta,
                    "video_frames": len(pts),
                    "video_fps": video_fps,
                    "parquet_sha256": entry_data["sha256"],
                    "video_sha256": entry_video["sha256"],
                }
                group_episodes.append(episode)
                episode_reports.append(
                    {key: value for key, value in episode.items() if key != "actions"}
                )
            for progress in QUERY_PROGRESS:
                query_indices = [
                    round((episode["length"] - 1) * progress) for episode in group_episodes
                ]
                options = [
                    episode["actions"][index].tolist()
                    for episode, index in zip(group_episodes, query_indices, strict=True)
                ]
                separation = numeric_option_separation(options)
                group_reports.append(
                    {
                        "split": split,
                        "prompt_sha256": group["prompt_sha256"],
                        "relative_progress": progress,
                        "episode_ids": [episode["episode_id"] for episode in group_episodes],
                        "query_indices": query_indices,
                        "candidate_vectors_distinct_at_tolerance": not bool(
                            separation["within_tolerance_pairs"]
                        ),
                        "option_separation": separation,
                    }
                )
    prompt_overlap = selected_prompt_hashes["train"] & selected_prompt_hashes["development"]
    media_overlap = {
        media_hash for media_hash, splits in file_hash_splits.items() if len(splits) > 1
    }
    frame_overlap = {
        content_hash for content_hash, splits in normalized_frame_splits.items() if len(splits) > 1
    }
    duplicate_choice_rows = sum(
        not row["candidate_vectors_distinct_at_tolerance"] for row in group_reports
    )
    report = {
        "schema_version": "robopro-action-probe-data-audit-v1",
        "status": "audit_complete",
        "dataset_revision": sample["revision"],
        "sample_manifest_sha256": hashlib.sha256(sample_bytes).hexdigest(),
        "download_receipt_sha256": sha256_file(receipt_path),
        "episode_count": len(episode_reports),
        "train_episode_count": sum(len(group["episodes"]) for group in sample["train"]),
        "development_episode_count": sum(len(group["episodes"]) for group in sample["development"]),
        "prompt_group_count": len(sample["train"]) + len(sample["development"]),
        "prompt_overlap_count": len(prompt_overlap),
        "media_file_hash_overlap_count": len(media_overlap),
        "normalized_32x32_frame_hash_overlap_count": len(frame_overlap),
        "candidate_group_query_count": len(group_reports),
        "duplicate_candidate_query_count": duplicate_choice_rows,
        "duplicate_candidate_query_rows": group_reports,
        "episodes": episode_reports,
        "cache_or_training_run_created": False,
        "official_evaluation_or_sealed_audit_accessed": False,
        "limitations": [
            "Exact-prompt separation does not establish physical scene separation.",
            "Action choices represent recorded commands, not success or optimality.",
            "Candidate options may be semantically multi-valid despite numeric uniqueness.",
            "The development partition is constructed from publisher TRAIN.",
        ],
    }
    if prompt_overlap or media_overlap or frame_overlap:
        raise ValueError("cross-partition identity/content overlap was detected")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(report, indent=2) + "\n").encode("utf-8")
    output_path.write_bytes(payload)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-manifest", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(
        args.sample_manifest.resolve(strict=True),
        args.split_manifest.resolve(strict=True),
        args.metadata_dir.resolve(strict=True),
        args.data_root.resolve(strict=True),
        args.output.resolve(),
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "episodes": report["episode_count"],
                "candidate_queries": report["candidate_group_query_count"],
                "duplicate_candidate_queries": report["duplicate_candidate_query_count"],
                "media_overlap": report["media_file_hash_overlap_count"],
                "normalized_frame_overlap": report["normalized_32x32_frame_hash_overlap_count"],
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
