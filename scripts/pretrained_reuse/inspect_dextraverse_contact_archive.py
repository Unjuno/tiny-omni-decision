"""Read-only hash, array, media-identity and target-support preflight for DextraVerse."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from scripts.pretrained_reuse.dextraverse_contact import (
    LABELS,
    build_contact_windows,
    inner_episode_partitions,
    outer_episode_folds,
)

EXPECTED_ARCHIVE_BYTES = 44_197_744_857
EXPECTED_ARCHIVE_SHA256 = "9f9d8d0e6a6eebac12f1bf63fae59bea46e9531db2b9921440aa553289d07aa4"
EXPECTED_FRAMES = 138_382
EXPECTED_EPISODES = 261
BIMANUAL_EPISODES = 3
MIN_USABLE_EPISODES = 200
MIN_VALID_QUERIES = 1_000
MIN_CLASS_QUERIES = 50
MIN_CLASS_EPISODES = 15


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_archive(archive_path: Path) -> dict[str, Any]:
    """Verify publisher hash and inspect only the fixed candidate's arrays."""
    import zarr
    from zarr.storage import ZipStore

    if archive_path.stat().st_size != EXPECTED_ARCHIVE_BYTES:
        raise ValueError("archive byte length does not match the pinned source")
    archive_sha256 = _sha256(archive_path)
    if archive_sha256 != EXPECTED_ARCHIVE_SHA256:
        raise ValueError("archive SHA-256 does not match publisher checksum")

    started = time.perf_counter()
    with ZipStore(str(archive_path), mode="r") as store:
        root = zarr.open_group(store=store, mode="r")
        arrays: dict[str, Any] = {}

        def collect(name: str, item: Any) -> None:
            if isinstance(item, zarr.Array):
                arrays[name] = item

        root.visititems(collect)
        required = (
            "meta/episode_ends",
            "data/timestamps",
            "data/left_tactile_data_fingerforce",
            "data/right_tactile_data_fingerforce",
            "data/camera_ego_rgb",
        )
        missing = [name for name in required if name not in arrays]
        if missing:
            raise ValueError(f"pinned README arrays are missing: {missing}")

        episode_ends = np.asarray(arrays["meta/episode_ends"][:], dtype=np.int64)
        timestamps = np.asarray(arrays["data/timestamps"][:], dtype=np.float64)
        if (episode_ends.shape != (EXPECTED_EPISODES,)
                or episode_ends[-1] != EXPECTED_FRAMES
                or timestamps.shape != (EXPECTED_FRAMES,)):
            raise ValueError("episode boundaries or frame count differ from pinned README")
        if np.any(np.diff(episode_ends) <= 0):
            raise ValueError("episode_ends must be strictly increasing")

        boundaries = np.concatenate(([0], episode_ends[:-1]))
        episode_rows: list[dict[str, Any]] = []
        for episode_id, (start, end) in enumerate(zip(boundaries, episode_ends, strict=True)):
            local_times = timestamps[int(start):int(end)]
            episode_rows.append({
                "episode_id": episode_id,
                "start_index": int(start),
                "end_index_exclusive": int(end),
                "frame_count": int(end - start),
                "duration_s": float(local_times[-1] - local_times[0]),
                "timestamp_strictly_increasing": bool(np.all(np.diff(local_times) > 0)),
            })

        label_counts = np.zeros(len(LABELS), dtype=np.int64)
        label_episode_sets = [set() for _ in LABELS]
        media_hash_to_episodes: dict[str, list[int]] = {}
        episode_folds = outer_episode_folds(list(range(EXPECTED_EPISODES)))
        usable_episode_ids: list[int] = []
        for episode_id, (start, end) in enumerate(zip(boundaries, episode_ends, strict=True)):
            row = episode_rows[episode_id]
            if not row["timestamp_strictly_increasing"]:
                row["status"] = "INVALID_TIMESTAMP_ORDER"
                continue
            start_i, end_i = int(start), int(end)
            local_times = timestamps[start_i:end_i]
            left_force = np.asarray(
                arrays["data/left_tactile_data_fingerforce"][start_i:end_i],
                dtype=np.float32,
            )
            right_force = np.asarray(
                arrays["data/right_tactile_data_fingerforce"][start_i:end_i],
                dtype=np.float32,
            )
            forces = np.concatenate((left_force, right_force), axis=1)
            video = np.asarray(arrays["data/camera_ego_rgb"][start_i:end_i])
            if video.shape != (end_i - start_i, 240, 320, 3):
                raise ValueError(f"unexpected head RGB shape for episode {episode_id}")
            video_valid = np.isfinite(video).all(axis=(1, 2, 3))
            video_valid &= np.any(video != 0, axis=(1, 2, 3))
            windows = build_contact_windows(local_times, forces, video_valid=video_valid)
            media_hash = hashlib.sha256(video.tobytes(order="C")).hexdigest()
            media_hash_to_episodes.setdefault(media_hash, []).append(episode_id)
            row["head_video_sha256"] = media_hash
            row["valid_query_count"] = len(windows)
            row["label_counts"] = {label: 0 for label in LABELS}
            for window in windows:
                label = LABELS[window.target]
                label_counts[window.target] += 1
                label_episode_sets[window.target].add(episode_id)
                row["label_counts"][label] += 1
            if windows:
                usable_episode_ids.append(episode_id)
                row["status"] = "USABLE"
            else:
                row["status"] = "NO_VALID_QUERY_WINDOWS"

        duplicate_hash_groups = {
            media_hash: episode_ids
            for media_hash, episode_ids in media_hash_to_episodes.items()
            if len(episode_ids) > 1
        }
        cross_fold_duplicates = {
            media_hash: episode_ids
            for media_hash, episode_ids in duplicate_hash_groups.items()
            if len({episode_folds[episode_id] for episode_id in episode_ids}) > 1
        }
        usable_folds = {i: episode_folds[i] for i in usable_episode_ids}
        fold_class_counts: dict[str, dict[str, int]] = {}
        fold_class_episodes: dict[str, dict[str, int]] = {}
        for fold in range(5):
            fold_class_counts[str(fold)] = {label: 0 for label in LABELS}
            fold_class_episodes[str(fold)] = {label: 0 for label in LABELS}
            for row in episode_rows:
                if row.get("status") != "USABLE" or usable_folds[row["episode_id"]] != fold:
                    continue
                for label, count in row["label_counts"].items():
                    fold_class_counts[str(fold)][label] += count
                    fold_class_episodes[str(fold)][label] += int(count > 0)

        class_support = {
            label: {
                "queries": int(label_counts[index]),
                "episodes": len(label_episode_sets[index]),
            }
            for index, label in enumerate(LABELS)
        }
        insufficient_classes = [
            label for label, support in class_support.items()
            if support["queries"] < MIN_CLASS_QUERIES
            or support["episodes"] < MIN_CLASS_EPISODES
        ]
        usable_query_count = int(label_counts.sum())
        usable = len(usable_episode_ids)
        invalid_time_episode_ids = [
            row["episode_id"] for row in episode_rows
            if not row["timestamp_strictly_increasing"]
        ]
        group_split_possible = usable >= MIN_USABLE_EPISODES
        if group_split_possible:
            for fold in range(5):
                train_ids = [i for i in usable_episode_ids if usable_folds[i] != fold]
                inner_episode_partitions(train_ids, outer_fold=fold)
        status = "PREFLIGHT_PASS" if (
            usable >= MIN_USABLE_EPISODES
            and usable_query_count >= MIN_VALID_QUERIES
            and not insufficient_classes
            and not cross_fold_duplicates
            and not invalid_time_episode_ids
        ) else "STOP_INSUFFICIENT_SUPPORT_OR_IDENTITY_OVERLAP"
        return {
            "format": "dextraverse_contact_archive_preflight_v1",
            "status": status,
            "archive_sha256": archive_sha256,
            "archive_bytes": archive_path.stat().st_size,
            "zarr_version": zarr.__version__,
            "array_shapes": {name: list(array.shape) for name, array in arrays.items()},
            "episode_count": len(episode_rows),
            "frame_count": len(timestamps),
            "invalid_timestamp_episode_ids": invalid_time_episode_ids,
            "bimanual_episodes_reported_by_source": BIMANUAL_EPISODES,
            "usable_episode_count": usable,
            "valid_query_count": usable_query_count,
            "class_support": class_support,
            "insufficient_classes": insufficient_classes,
            "outer_fold_class_query_counts": fold_class_counts,
            "outer_fold_class_episode_counts": fold_class_episodes,
            "outer_fold_episode_counts": {
                str(fold): sum(value == fold for value in usable_folds.values())
                for fold in range(5)
            },
            "duplicate_head_video_hashes": duplicate_hash_groups,
            "cross_fold_duplicate_head_video_hashes": cross_fold_duplicates,
            "episode_rows": episode_rows,
            "elapsed_seconds": time.perf_counter() - started,
            "model_fit_or_quality_evaluation": False,
            "sealed_audit_accessed": False,
            "decision": (
                "Proceed to frozen features and grouped OOF only if all gates pass."
                if status == "PREFLIGHT_PASS"
                else "Do not fit, collapse classes, or relax the frozen rules."
            ),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite preflight report: {args.output}")
    report = inspect_archive(args.archive.resolve(strict=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        key: report.get(key)
        for key in ("status", "usable_episode_count", "valid_query_count", "class_support")
        if key in report
    }, indent=2))


if __name__ == "__main__":
    main()
