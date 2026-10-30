"""Build deterministic synchronized wrist-video/IMU windows from a frozen manifest.

Optional MCAP/Protobuf dependencies are imported only when ``build`` is called.
The command refuses to overwrite an existing output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import struct
from pathlib import Path
from typing import Any

import numpy as np

WINDOW_NS = 2_000_000_000
STRIDE_NS = 1_000_000_000
IMU_POINTS = 128
VIDEO_TOPIC = "/wrist_left/image/compressed"
IMU_TOPIC = "/wrist_left/imu/data"


def _decode_imu(payload: bytes) -> tuple[int, tuple[float, ...], tuple[float, ...]]:
    """Decode the fixed ROS2 CDR fields needed for timestamp, acceleration and gyro."""
    if payload[:4] != b"\x00\x01\x00\x00":
        raise ValueError("unexpected CDR encapsulation")
    offset = 4
    seconds = struct.unpack_from("<i", payload, offset)[0]
    offset += 4
    nanos = struct.unpack_from("<I", payload, offset)[0]
    offset += 4

    def align(position: int, alignment: int) -> int:
        return 4 + ((position - 4 + alignment - 1) // alignment) * alignment

    offset = align(offset, 4)
    string_size = struct.unpack_from("<I", payload, offset)[0]
    offset += 4 + string_size
    if string_size < 1 or payload[offset - 1] != 0:
        raise ValueError("invalid CDR frame id")
    offset = align(offset, 8) + 4 * 8 + 9 * 8
    gyro = struct.unpack_from("<3d", payload, offset)
    offset += 3 * 8 + 9 * 8
    acceleration = struct.unpack_from("<3d", payload, offset)
    offset += 3 * 8 + 9 * 8
    if offset != len(payload) or not all(
        math.isfinite(value) for value in (*gyro, *acceleration)
    ):
        raise ValueError("invalid or non-finite CDR IMU payload")
    return seconds * 1_000_000_000 + nanos, tuple(acceleration), tuple(gyro)


def _video_message_class(schema_data: bytes):
    from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

    descriptor_set = descriptor_pb2.FileDescriptorSet()
    descriptor_set.ParseFromString(schema_data)
    pool = descriptor_pool.DescriptorPool()
    for descriptor in descriptor_set.file:
        pool.Add(descriptor)
    descriptor = pool.FindMessageTypeByName("foxglove.CompressedVideo")
    return message_factory.GetMessageClass(descriptor)


def choose_frame_indices(frame_times_ns: list[int], start_ns: int, end_ns: int) -> list[int]:
    """Return eight approximately uniform frame indices in a half-open interval."""
    candidates = [index for index, stamp in enumerate(frame_times_ns) if start_ns <= stamp < end_ns]
    if len(candidates) < 8:
        return []
    positions = np.rint(np.linspace(0, len(candidates) - 1, 8)).astype(int)
    return [candidates[int(position)] for position in positions]


def resample_imu(
    timestamps_ns: list[int], values: np.ndarray, start_ns: int, points: int = IMU_POINTS
) -> np.ndarray:
    """Linearly resample six IMU channels inside a two-second interval."""
    if values.shape != (len(timestamps_ns), 6) or len(timestamps_ns) < 2:
        raise ValueError("IMU values must have shape [timestamps, 6]")
    stamps = np.asarray(timestamps_ns, dtype=np.int64)
    if np.any(np.diff(stamps) <= 0):
        raise ValueError("IMU timestamps must be strictly increasing")
    targets = start_ns + np.arange(points, dtype=np.float64) * WINDOW_NS / points
    if targets[0] < stamps[0] or targets[-1] > stamps[-1]:
        raise ValueError("IMU samples do not cover the requested interval")
    result = np.stack(
        [np.interp(targets, stamps.astype(np.float64), values[:, channel]) for channel in range(6)]
    ).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("resampled IMU contains non-finite values")
    return result


def make_candidates(
    chunk_id: str, query_index: int, count: int, seed: int = 17
) -> tuple[list[int], int]:
    """Create one matching option and three same-recording negatives, then shuffle."""
    distractors = [index for index in range(count) if abs(index - query_index) >= 3]
    if len(distractors) < 3:
        raise ValueError("at least three non-adjacent distractor windows are required")
    key = hashlib.sha256(f"{seed}:{chunk_id}:{query_index}".encode()).digest()
    rng = random.Random(int.from_bytes(key[:8], "big"))
    options = [query_index, *rng.sample(distractors, 3)]
    rng.shuffle(options)
    return options, options.index(query_index)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build(manifest_path: Path, output: Path, seed: int = 17) -> dict[str, Any]:
    """Read only wrist-left video/IMU topics and emit an immutable corpus directory."""
    from mcap.reader import make_reader

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=False)
    all_windows: list[np.ndarray] = []
    examples: dict[str, list[dict[str, Any]]] = {"train": [], "development": []}
    inventory: list[dict[str, Any]] = []
    files = manifest.get("files", [])
    if not files:
        raise ValueError("manifest has no selected files")

    for item in files:
        source = Path(item["path"])
        if source.stat().st_size != item["bytes"] or _sha256(source) != item["sha256"]:
            raise ValueError(f"source size/SHA-256 mismatch: {item['chunk_id']}")
        video_times: list[int] = []
        imu_times: list[int] = []
        imu_values: list[tuple[float, ...]] = []
        video_class = None
        with source.open("rb") as stream:
            reader = make_reader(stream, validate_crcs=True)
            for schema, channel, message in reader.iter_messages(topics=[VIDEO_TOPIC, IMU_TOPIC]):
                if channel.topic == VIDEO_TOPIC:
                    if schema is None or schema.name != "foxglove.CompressedVideo":
                        raise ValueError("unexpected compressed-video schema")
                    if video_class is None:
                        video_class = _video_message_class(schema.data)
                    decoded = video_class()
                    decoded.ParseFromString(message.data)
                    if decoded.format.lower() != "h264":
                        raise ValueError("expected H.264 video")
                    video_times.append(
                        decoded.timestamp.seconds * 1_000_000_000 + decoded.timestamp.nanos
                    )
                else:
                    if schema is None or schema.name != "sensor_msgs/msg/Imu":
                        raise ValueError("unexpected IMU schema")
                    stamp, acceleration, angular_velocity = _decode_imu(message.data)
                    imu_times.append(stamp)
                    imu_values.append((*acceleration, *angular_velocity))

        if not video_times or not imu_times:
            raise ValueError(f"empty streams: {item['chunk_id']}")
        if np.any(np.diff(video_times) <= 0) or np.any(np.diff(imu_times) <= 0):
            raise ValueError(f"non-monotonic stream timestamps: {item['chunk_id']}")
        median_period = int(np.median(np.diff(np.asarray(video_times, dtype=np.int64))))
        common_start = max(video_times[0], imu_times[0])
        common_end = min(video_times[-1] + median_period, imu_times[-1])
        query_count = max(0, (common_end - common_start - WINDOW_NS) // STRIDE_NS + 1)

        starts: list[int] = []
        frame_indices: list[list[int]] = []
        for query_index in range(int(query_count)):
            start = common_start + query_index * STRIDE_NS
            chosen = choose_frame_indices(video_times, start, start + WINDOW_NS)
            if chosen:
                starts.append(start)
                frame_indices.append(chosen)
        if len(starts) < 4:
            raise ValueError(f"too few valid windows: {item['chunk_id']}")

        local_indices: list[int] = []
        imu_array = np.asarray(imu_values, dtype=np.float64)
        for start in starts:
            local_indices.append(len(all_windows))
            all_windows.append(resample_imu(imu_times, imu_array, start))

        for query_index, (start, frames) in enumerate(zip(starts, frame_indices, strict=True)):
            try:
                local_options, target = make_candidates(
                    item["chunk_id"], query_index, len(starts), seed
                )
            except ValueError:
                continue
            choices = [local_indices[index] for index in local_options]
            sample_id = hashlib.sha256(
                f"{seed}:{item['chunk_id']}:{query_index}".encode()
            ).hexdigest()[:24]
            examples[item["split"]].append(
                {
                    "sample_id": sample_id,
                    "split": item["split"],
                    "group_id": item["group_id"],
                    "category": item["category"],
                    "chunk_id": item["chunk_id"],
                    "query_index": query_index,
                    "query_interval_ns": [start, start + WINDOW_NS],
                    "video_frame_indices": frames,
                    "candidate_window_indices": choices,
                    "target_option_index": target,
                }
            )
        inventory.append(
            {
                "chunk_id": item["chunk_id"],
                "group_id": item["group_id"],
                "split": item["split"],
                "video_frames": len(video_times),
                "imu_samples": len(imu_times),
                "windows": len(starts),
            }
        )

    train_groups = {row["group_id"] for row in inventory if row["split"] == "train"}
    dev_groups = {row["group_id"] for row in inventory if row["split"] == "development"}
    if train_groups & dev_groups:
        raise ValueError("train/development session overlap")
    for split_rows in examples.values():
        split_rows.sort(key=lambda row: row["sample_id"])

    array = np.stack(all_windows)
    with (output / "imu-windows.npy").open("xb") as stream:
        np.save(stream, array, allow_pickle=False)
    for split, rows in examples.items():
        sample_path = output / f"{split}-samples.jsonl"
        with sample_path.open("x", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, separators=(",", ":")) + "\n")

    report = {
        "format": "egoviz-vs-sync-window-corpus-v1",
        "seed": seed,
        "window_seconds": 2,
        "stride_seconds": 1,
        "video_frames_per_window": 8,
        "imu_points_per_window": IMU_POINTS,
        "imu_channels": ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"],
        "train_examples": len(examples["train"]),
        "development_examples": len(examples["development"]),
        "train_groups": len(train_groups),
        "development_groups": len(dev_groups),
        "group_overlap": len(train_groups & dev_groups),
        "unique_imu_windows": len(all_windows),
        "tensor_shape": list(array.shape),
        "publisher_action_labels_read": False,
        "sealed_audit_accessed": False,
        "inventory": inventory,
        "sha256": {
            name: _sha256(output / name)
            for name in ("imu-windows.npy", "train-samples.jsonl", "development-samples.jsonl")
        },
    }
    with (output / "window-corpus.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    arguments = parser.parse_args()
    print(json.dumps(build(arguments.manifest, arguments.output, arguments.seed), indent=2))


if __name__ == "__main__":
    main()
