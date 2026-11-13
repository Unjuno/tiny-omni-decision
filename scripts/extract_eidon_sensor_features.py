"""Extract pinned, quaternion-only Eidon features for train or development."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from scripts.eidon_range_io import HTTPRangeReader
from tiny_omni_decision.eidon_pairing import relative_orientation_window_features

VIDEO_REVISION = "47c55ccc6d0308894ab482009c6fa91569057a08"
IMU_REVISION = "97d4441a7e915d2ad4b5988a43ebddd79a542aea"
IMU_COLUMNS = (
    "recording_id", "time_ms", "slot", "quat_x", "quat_y", "quat_z", "quat_w"
)


def midpoint_window(video_duration_seconds: float, sensor_first_timestamp_ms: float,
                    window_seconds: float = 5.0) -> tuple[float, float]:
    if (not np.isfinite(video_duration_seconds) or video_duration_seconds <= 0
            or not np.isfinite(sensor_first_timestamp_ms) or sensor_first_timestamp_ms < 0
            or not np.isfinite(window_seconds) or window_seconds <= 0
            or window_seconds > video_duration_seconds):
        raise ValueError("finite positive durations and a contained window are required")
    start = sensor_first_timestamp_ms + (video_duration_seconds - window_seconds) * 500.0
    return start, start + window_seconds * 1000.0


def _load_split(manifest_path: Path, metadata_path: Path, split: str) -> list[dict]:
    if split not in {"train", "development"}:
        raise ValueError("only train and development are allowed")
    if manifest_path.name != "train-development-manifest-private.json":
        raise ValueError("only the frozen train/development manifest is allowed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("video_revision") != VIDEO_REVISION:
        raise ValueError("unexpected pinned video revision")
    rows = manifest.get(split)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"frozen {split} split is missing")
    metadata = pq.read_table(metadata_path, columns=[
        "recording_id", "qc_status", "n_slots", "has_chest", "duration_seconds",
        "video_duration_seconds",
    ]).to_pydict()
    by_id = {int(recording_id): index for index, recording_id
             in enumerate(metadata["recording_id"])}
    enriched = []
    for row in rows:
        index = by_id.get(int(row["recording_id"]))
        if index is None:
            raise ValueError("frozen split recording is absent from pinned metadata")
        if (metadata["qc_status"][index] != "valid" or metadata["n_slots"][index] != 7
                or not metadata["has_chest"][index]):
            raise ValueError("frozen split contains an incomplete or invalid sensor rig")
        enriched.append(row | {
            "imu_duration_seconds": float(metadata["duration_seconds"][index]),
        })
    return enriched


def extract_split(manifest_path: Path, metadata_path: Path, shard_index_path: Path, split: str,
                  output_dir: Path) -> dict:
    rows = _load_split(manifest_path, metadata_path, split)
    shard_index = json.loads(shard_index_path.read_text(encoding="utf-8"))
    by_shard: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        matches = [item for item in shard_index
                   if item["first_recording_id"] <= row["recording_id"]
                   <= item["last_recording_id"]]
        if len(matches) != 1:
            raise ValueError("recording does not map to exactly one pinned IMU shard")
        by_shard[matches[0]["shard"]].append(row)

    output_dir.mkdir(parents=True, exist_ok=False)
    features: dict[int, np.ndarray] = {}
    provenance: dict[int, dict] = {}
    invalid: dict[int, str] = {}
    started = time.perf_counter()
    totals = {"range_bytes": 0, "range_requests": 0, "row_groups_read": 0}
    for shard_name, shard_rows in sorted(by_shard.items()):
        recording_ids = {row["recording_id"] for row in shard_rows}
        print(f"Reading shard {shard_name}; target recordings={len(recording_ids)}", flush=True)
        url = (f"https://huggingface.co/datasets/eidon-ai/tracker-pov-imu/resolve/"
               f"{IMU_REVISION}/{shard_name}")
        with HTTPRangeReader(url, max_bytes=512 * 1024 * 1024,
                             block_size=8 * 1024 * 1024, max_cached_blocks=4) as source:
            parquet = pq.ParquetFile(source)
            if tuple(parquet.schema.names) != (
                "recording_id", "time_ms", "slot", "quat_x", "quat_y", "quat_z",
                "quat_w", "accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y",
                "gyro_z", "mag_x", "mag_y", "mag_z"
            ):
                raise ValueError("pinned IMU shard schema changed")
            for group_idx in range(parquet.metadata.num_row_groups):
                stats = parquet.metadata.row_group(group_idx).column(0).statistics
                if stats is None:
                    raise ValueError("recording_id row-group statistics are required")
                group_recordings = [recording_id for recording_id in recording_ids
                                    if stats.min <= recording_id <= stats.max]
                if not group_recordings:
                    continue
                table = parquet.read_row_group(group_idx, columns=list(IMU_COLUMNS))
                totals["row_groups_read"] += 1
                values = table.to_pydict()
                ids = np.asarray(values["recording_id"])
                for recording_id in group_recordings:
                    mask = ids == recording_id
                    if not mask.any() or recording_id in features:
                        continue
                    times = np.asarray(values["time_ms"], dtype=np.float64)[mask]
                    slots = np.asarray(values["slot"])[mask]
                    quaternions = np.column_stack([
                        np.asarray(values[name], dtype=np.float64)[mask]
                        for name in ("quat_x", "quat_y", "quat_z", "quat_w")
                    ])
                    ordered = np.unique(times)
                    if len(ordered) < 8 or np.any(np.diff(ordered) <= 0):
                        raise ValueError("sensor timestamps are insufficient or unordered")
                    duration = float(next(r["video_duration_seconds"] for r in shard_rows
                                          if r["recording_id"] == recording_id))
                    sensor_start = float(ordered[0])
                    window_start, window_end = midpoint_window(duration, sensor_start)
                    # Source documentation states video and IMU capture share the
                    # app clock to within a few milliseconds; the selected window
                    # is centered in the video using the first observed IMU time.
                    try:
                        features[recording_id] = relative_orientation_window_features(
                            times, slots, quaternions, window_start_ms=window_start,
                            window_end_ms=window_end, bins=8
                        )
                    except ValueError as exc:
                        in_window = ((times >= window_start) & (times < window_end))
                        norms = np.linalg.norm(quaternions[in_window], axis=1)
                        if not len(norms):
                            norm_range = "empty-window"
                        else:
                            invalid_count = int(np.count_nonzero(np.abs(norms - 1) > 1e-3))
                            norm_range = (
                                f"{norms.min():.8f}..{norms.max():.8f}; "
                                f"invalid_count={invalid_count}"
                            )
                        invalid[recording_id] = f"{exc}; norm_range={norm_range}"
                        continue
                    provenance[recording_id] = {
                        "asset_sha256": next(r["asset_sha256"] for r in shard_rows
                                             if r["recording_id"] == recording_id),
                        "contributor_id": next(r["contributor_id"] for r in shard_rows
                                                if r["recording_id"] == recording_id),
                        "task_type": next(r["task_type"] for r in shard_rows
                                          if r["recording_id"] == recording_id),
                        "split": split,
                        "video_duration_seconds": duration,
                        "imu_first_timestamp_ms": sensor_start,
                        "imu_last_timestamp_ms": float(ordered[-1]),
                        "selected_window_ms": [window_start, window_end],
                        "rows_in_recording": int(mask.sum()),
                        "unique_timestamps": int(len(ordered)),
                    }
                if recording_ids.issubset(features.keys() | invalid.keys()):
                    break
            totals["range_bytes"] += source.bytes_read
            totals["range_requests"] += source.range_requests
            found_in_shard = sum(row["recording_id"] in features for row in shard_rows)
            print(
                f"Finished shard {shard_name}; records={found_in_shard}; "
                f"range_bytes={source.bytes_read}; requests={source.range_requests}",
                flush=True,
            )

    groups: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["contributor_id"], row["task_type"])].append(row)
    if any(len(members) != 4 for members in groups.values()):
        raise ValueError("frozen split does not contain four-choice contributor/task groups")
    quarantined_groups = {
        key for key, members in groups.items()
        if any(member["recording_id"] in invalid for member in members)
    }
    usable_rows = sorted(
        (row for row in rows
         if (row["contributor_id"], row["task_type"]) not in quarantined_groups),
        key=lambda row: row["recording_id"],
    )
    if not usable_rows:
        raise ValueError("no complete contributor/task group passed sensor validation")
    if set(features) != {row["recording_id"] for row in rows} - set(invalid):
        raise ValueError("sensor extraction omitted records without an explicit validation error")
    for members in groups.values():
        if any(member["recording_id"] in invalid for member in members):
            for member in members:
                features.pop(member["recording_id"], None)
                provenance.pop(member["recording_id"], None)
    ordered_rows = usable_rows
    ordered_features = np.stack([features[row["recording_id"]] for row in ordered_rows])
    if (ordered_features.shape != (len(ordered_rows), 8, 54)
            or not np.isfinite(ordered_features).all()):
        raise ValueError("unexpected or non-finite sensor feature tensor")
    np.savez_compressed(
        output_dir / "features.npz", recording_ids=np.asarray(
            [row["recording_id"] for row in ordered_rows], dtype=np.int64
        ), values=ordered_features
    )
    records_path = output_dir / "records.json"
    records_path.write_text(json.dumps(
        [provenance[row["recording_id"]] | {"recording_id": row["recording_id"]}
         for row in ordered_rows], separators=(",", ":")
    ) + "\n", encoding="utf-8")
    quarantine_path = output_dir / "quarantine-private.json"
    quarantine_path.write_text(json.dumps({
        "invalid_recordings": invalid,
        "quarantined_groups": [
            {"contributor_id": key[0], "task_type": key[1],
             "recording_ids": [member["recording_id"] for member in groups[key]]}
            for key in sorted(quarantined_groups)
        ],
    }, indent=2) + "\n", encoding="utf-8")
    data_hash = hashlib.sha256((output_dir / "features.npz").read_bytes()).hexdigest()
    records_hash = hashlib.sha256(records_path.read_bytes()).hexdigest()
    result = {
        "split": split, "candidate_count": len(rows), "usable_count": len(ordered_rows),
        "quarantined_group_count": len(quarantined_groups),
        "quarantined_record_count": len(rows) - len(ordered_rows),
        "shape": list(ordered_features.shape),
        "source_video_revision": VIDEO_REVISION, "source_imu_revision": IMU_REVISION,
        "feature_sha256": data_hash, "records_sha256": records_hash,
        "elapsed_seconds": round(time.perf_counter() - started, 3), **totals,
        "sealed_audit_read": False,
        "alignment_assumption": (
            "shared app clock; midpoint 5-second window from first IMU timestamp"
        ),
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--shard-index", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "development"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(extract_split(args.manifest, args.metadata, args.shard_index, args.split,
                                   args.output_dir), sort_keys=True))


if __name__ == "__main__":
    main()
