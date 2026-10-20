"""Build 1-second causal feature sequences from pinned HoloAssist synchronized IMU files."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from scripts.audit_holoassist_metadata import sha256_file

SENSORS = ("Accelerometer", "Gyroscope", "Magnetometer")
WINDOW_SECONDS = 30
EXPECTED_RATE_HZ = 30.0


def add_sensor_rows(path: Path, *, duration: float, sensor_index: int, totals, squares, counts):
    """Accumulate one sensor stream into one-second bins; reject malformed or noncausal rows."""
    previous_seconds = -math.inf
    previous_timestamp = -1
    row_count = 0
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        for line_number, line in enumerate(source, start=1):
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) != 5:
                raise ValueError(f"expected five tab-separated columns at {path}:{line_number}")
            try:
                seconds = float(parts[0])
                timestamp = int(parts[1])
                values = np.asarray([float(value) for value in parts[2:]], dtype=np.float64)
            except ValueError as error:
                raise ValueError(f"invalid numeric IMU row at {path}:{line_number}") from error
            if (
                not math.isfinite(seconds)
                or seconds < 0
                or seconds <= previous_seconds
                or timestamp <= previous_timestamp
                or not np.isfinite(values).all()
            ):
                raise ValueError(f"non-finite or non-monotone IMU row at {path}:{line_number}")
            if seconds > duration + 1.0:
                raise ValueError(f"IMU timestamp exceeds session duration: {path}:{line_number}")
            previous_seconds = seconds
            previous_timestamp = timestamp
            row_count += 1
            if seconds > duration:
                continue
            # Right-closed bins match the action-label rule (start < event_end <= end).
            bin_index = max(0, math.ceil(seconds) - 1)
            offset = sensor_index * 3
            totals[bin_index, offset : offset + 3] += values
            squares[bin_index, offset : offset + 3] += values * values
            counts[bin_index, sensor_index] += 1
    if row_count == 0:
        raise ValueError(f"empty IMU sensor file: {path}")
    return row_count


def build_session_features(session_files: dict[str, Path], *, duration: float) -> np.ndarray:
    """Return [ceil(duration), 21] mean/std/coverage features with 1-second bins."""
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("session duration must be finite and positive")
    if set(session_files) != set(SENSORS):
        raise ValueError("each session requires accelerometer, gyroscope, and magnetometer files")
    bin_count = math.ceil(duration)
    totals = np.zeros((bin_count, 9), dtype=np.float64)
    squares = np.zeros((bin_count, 9), dtype=np.float64)
    counts = np.zeros((bin_count, 3), dtype=np.int64)
    for sensor_index, sensor in enumerate(SENSORS):
        add_sensor_rows(
            session_files[sensor],
            duration=duration,
            sensor_index=sensor_index,
            totals=totals,
            squares=squares,
            counts=counts,
        )
    channel_counts = np.repeat(counts, 3, axis=1)
    means = np.divide(totals, channel_counts, out=np.zeros_like(totals), where=channel_counts > 0)
    variances = (
        np.divide(squares, channel_counts, out=np.zeros_like(squares), where=channel_counts > 0)
        - means * means
    )
    stds = np.sqrt(np.maximum(variances, 0.0))
    coverage = np.clip(counts / EXPECTED_RATE_HZ, 0.0, 1.0)
    features = np.concatenate([means, stds, coverage], axis=1).astype(np.float32)
    if features.shape != (bin_count, 21) or not np.isfinite(features).all():
        raise ValueError("invalid non-finite IMU feature matrix")
    return features


def pool_temporal_features(features: np.ndarray, start: float, end: float) -> np.ndarray:
    """Flatten the fixed 30-bin window while respecting [start, end) boundaries."""
    if (
        features.ndim != 2
        or features.shape[1] != 21
        or not math.isfinite(start)
        or not math.isfinite(end)
        or start < 0
        or end <= start
        or end - start > WINDOW_SECONDS
        or abs(start - round(start)) > 1e-8
    ):
        raise ValueError("invalid temporal feature window")
    first = int(start)
    last = min(int(math.ceil(end)), first + WINDOW_SECONDS, len(features))
    result = np.zeros((WINDOW_SECONDS, 21), dtype=np.float32)
    width = last - first
    if width > 0:
        result[:width] = features[first:last]
    return result


def build_features(
    windows_dir: Path, imu_root: Path, manifest_path: Path, output_dir: Path
) -> dict:
    """Verify all acquired data, build aligned train/validation feature archives, and receipt."""
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite IMU feature output: {output_dir}")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if len(manifest.get("files", [])) != 3918:
        raise ValueError("complete 3,918-file selected IMU manifest is required")
    media_files = {}
    for row in manifest["files"]:
        key = (str(row["split"]), str(row["session_id"]), str(row["sensor"]))
        if key in media_files:
            raise ValueError(f"duplicate IMU manifest row: {key}")
        path = imu_root / key[0] / key[1] / f"{key[2]}_sync.txt"
        if (
            not path.is_file()
            or path.stat().st_size != int(row["size_bytes"])
            or sha256_file(path) != row["sha256"]
        ):
            raise ValueError(f"IMU file failed manifest integrity check: {path}")
        media_files[key] = path

    window_rows: dict[str, list[dict]] = {}
    for split in ("train", "validation"):
        path = windows_dir / f"{split}.jsonl"
        receipt = json.loads((windows_dir / "receipt.json").read_text(encoding="utf-8"))
        if sha256_file(path) != receipt[f"{split}_jsonl_sha256"]:
            raise ValueError(f"frozen {split} window file hash mismatch")
        window_rows[split] = [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line
        ]

    session_splits = {
        (str(row["session_id"]), split) for split, rows in window_rows.items() for row in rows
    }
    if len({session for session, _ in session_splits}) != len(session_splits):
        raise ValueError("a HoloAssist session occurs in more than one window split")
    manifest_sessions = {(session, split) for split, session, _ in media_files}
    if session_splits != manifest_sessions:
        raise ValueError("IMU and frozen temporal-window session splits differ")

    session_arrays: dict[str, np.ndarray] = {}
    for session, split in sorted(session_splits):
        rows = [row for row in window_rows[split] if row["session_id"] == session]
        durations = {float(row["video_duration_seconds"]) for row in rows}
        if len(durations) != 1:
            raise ValueError(f"inconsistent duration for session {session}")
        files = {sensor: media_files[(split, session, sensor)] for sensor in SENSORS}
        session_arrays[session] = build_session_features(files, duration=durations.pop())

    output_dir.mkdir(parents=True)
    split_outputs = {}
    for split, rows in window_rows.items():
        sequences = []
        metadata = []
        for row in rows:
            session = str(row["session_id"])
            start = float(row["window_start_seconds"])
            end = float(row["window_end_seconds"])
            feature = pool_temporal_features(session_arrays[session], start, end)
            sequences.append(feature)
            metadata.append(
                {
                    key: row[key]
                    for key in (
                        "id",
                        "split",
                        "session_id",
                        "window_index",
                        "window_start_seconds",
                        "window_end_seconds",
                        "target",
                        "loss_mask",
                        "media_sha256",
                        "task_type",
                    )
                }
            )
        target = output_dir / f"{split}.npz"
        np.savez_compressed(
            target,
            sample_id=np.asarray([row["id"] for row in rows], dtype=str),
            features=np.asarray(sequences, dtype=np.float32),
            flattened_features=np.asarray(
                [sequence.reshape(-1) for sequence in sequences], dtype=np.float32
            ),
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
        split_outputs[split] = {
            "path": str(target),
            "sha256": sha256_file(target),
            "sample_count": len(rows),
            "unique_session_count": len({str(row["session_id"]) for row in rows}),
            "unique_sample_id_order_sha256": hashlib.sha256(
                "\n".join(str(row["id"]) for row in rows).encode("utf-8")
            ).hexdigest(),
            "sequence_shape": [len(rows), WINDOW_SECONDS, 21],
            "flattened_feature_shape": [len(rows), WINDOW_SECONDS * 21],
        }
    receipt = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "feature_definition": (
            "right-closed one-second bins aligned to (window_start, window_end]; "
            "9-axis mean, 9-axis standard deviation, 3 sensor coverage values; "
            "30 bins per window, with the first bin including the recording's t=0 sample"
        ),
        "future_sensor_samples_used": False,
        "sealed_audit_loaded": False,
        "window_receipt_sha256": sha256_file(windows_dir / "receipt.json"),
        "imu_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "splits": split_outputs,
        "files_verified": len(media_files),
        "unique_sessions": len(session_arrays),
    }
    (output_dir / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows-dir", type=Path, required=True)
    parser.add_argument("--imu-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = build_features(args.windows_dir, args.imu_root, args.manifest, args.output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
