"""Bounded EgoViz dev-split wrist-video/IMU decode and timing smoke.

This verifies only one source record's bytes, session assignment, codec and
timestamp compatibility. It does not read action-label payloads, train a
model, or calculate model quality.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import io
import json
import math
import statistics
import struct
from pathlib import Path

import av
import pandas as pd
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
from mcap.reader import make_reader

INDEX_SHA256 = "c83126637cc24a7ac2b3600724b50cf9c53966b9b098acc57d5534e14ade1074"
SPLIT_PREFIX = "seed17:egoviz-session-split-v1:"
CHUNK_ID = "7f2a80d98e54f0559893a5bf5cee0e5c"
VIDEO_TOPIC = "/wrist_left/image/compressed"
IMU_TOPIC = "/wrist_left/imu/data"


def _align_cdr(offset: int, alignment: int) -> int:
    # CDR alignment starts after the four-byte encapsulation header.
    origin = 4
    return origin + ((offset - origin + alignment - 1) // alignment) * alignment


def _read_cdr_string(payload: bytes, offset: int) -> tuple[str, int]:
    offset = _align_cdr(offset, 4)
    size = struct.unpack_from("<I", payload, offset)[0]
    offset += 4
    raw = payload[offset : offset + size]
    if len(raw) != size or not raw.endswith(b"\0"):
        raise ValueError("invalid CDR string")
    return raw[:-1].decode("utf-8", errors="strict"), offset + size


def _read_cdr_f64s(payload: bytes, offset: int, count: int) -> tuple[tuple[float, ...], int]:
    offset = _align_cdr(offset, 8)
    values = struct.unpack_from("<" + "d" * count, payload, offset)
    return values, offset + 8 * count


def _decode_imu(payload: bytes) -> tuple[int, str, tuple[float, ...], tuple[float, ...]]:
    if payload[:4] != b"\x00\x01\x00\x00":
        raise ValueError("unexpected CDR encapsulation")
    offset = 4
    sec = struct.unpack_from("<i", payload, offset)[0]
    offset += 4
    nsec = struct.unpack_from("<I", payload, offset)[0]
    offset += 4
    frame_id, offset = _read_cdr_string(payload, offset)
    quat, offset = _read_cdr_f64s(payload, offset, 4)
    orientation_cov, offset = _read_cdr_f64s(payload, offset, 9)
    gyro, offset = _read_cdr_f64s(payload, offset, 3)
    gyro_cov, offset = _read_cdr_f64s(payload, offset, 9)
    accel, offset = _read_cdr_f64s(payload, offset, 3)
    accel_cov, offset = _read_cdr_f64s(payload, offset, 9)
    if offset != len(payload):
        raise ValueError("unexpected trailing CDR data")
    if not all(
        math.isfinite(value)
        for value in quat + orientation_cov + gyro + gyro_cov + accel + accel_cov
    ):
        raise ValueError("non-finite IMU value")
    return sec * 1_000_000_000 + nsec, frame_id, tuple(accel), tuple(gyro)


def _video_message_class(schema_data: bytes):
    descriptor_set = descriptor_pb2.FileDescriptorSet()
    descriptor_set.ParseFromString(schema_data)
    pool = descriptor_pool.DescriptorPool()
    for descriptor in descriptor_set.file:
        pool.Add(descriptor)
    descriptor = pool.FindMessageTypeByName("foxglove.CompressedVideo")
    return message_factory.GetMessageClass(descriptor)


def _percentile95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[int(0.95 * (len(ordered) - 1))]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_receipt(index_path: Path, split_path: Path, mcap_path: Path) -> dict[str, object]:
    index_hash = _sha256(index_path)
    if index_hash != INDEX_SHA256:
        raise ValueError(f"unexpected index hash: {index_hash}")
    index = pd.read_parquet(index_path)
    rows = index[index["chunk_id"] == CHUNK_ID]
    if len(rows) != 1:
        raise ValueError("selected chunk must occur exactly once in the index")
    row = rows.iloc[0]
    group_id = str(row["group_id"])
    if mcap_path.stat().st_size != int(row["mcap_bytes"]):
        raise ValueError("downloaded MCAP size does not match frozen index")

    split = json.loads(split_path.read_text(encoding="utf-8"))
    if split.get("index_sha256") != INDEX_SHA256 or split.get("seed") != 17:
        raise ValueError("split receipt does not match the expected frozen index/seed")
    groups = sorted(
        index["group_id"].astype(str).unique(),
        key=lambda group: hashlib.sha256((SPLIT_PREFIX + group).encode()).hexdigest(),
    )
    development_groups = set(groups[: round(len(groups) * 0.2)])
    expected_development_hashes = {
        hashlib.sha256(group.encode()).hexdigest() for group in development_groups
    }
    expected_train_hashes = {
        hashlib.sha256(group.encode()).hexdigest() for group in set(groups) - development_groups
    }
    if expected_development_hashes != set(split.get("development_group_sha256", [])):
        raise ValueError("development group assignment differs from the frozen split receipt")
    if expected_train_hashes != set(split.get("train_group_sha256", [])):
        raise ValueError("train group assignment differs from the frozen split receipt")
    if group_id not in development_groups:
        raise ValueError("selected source group is not in the deterministic development split")

    topic_counts: dict[str, int] = {}
    with mcap_path.open("rb") as stream:
        for _schema, channel, _message in make_reader(stream, validate_crcs=True).iter_messages():
            topic_counts[channel.topic] = topic_counts.get(channel.topic, 0) + 1
    video_class = None
    video_timestamps: list[int] = []
    video_packets: list[bytes] = []
    video_frame_ids: set[str] = set()
    video_formats: set[str] = set()
    imu_timestamps: list[int] = []
    imu_frame_ids: set[str] = set()
    accelerations: list[tuple[float, ...]] = []
    angular_velocities: list[tuple[float, ...]] = []

    with mcap_path.open("rb") as stream:
        for schema, _channel, message in make_reader(stream, validate_crcs=True).iter_messages(
            topics=[VIDEO_TOPIC]
        ):
            if schema is None or schema.name != "foxglove.CompressedVideo":
                raise ValueError("unexpected wrist video schema")
            if video_class is None:
                video_class = _video_message_class(schema.data)
            decoded = video_class()
            decoded.ParseFromString(message.data)
            video_timestamps.append(
                decoded.timestamp.seconds * 1_000_000_000 + decoded.timestamp.nanos
            )
            video_frame_ids.add(decoded.frame_id)
            video_formats.add(decoded.format.lower())
            video_packets.append(bytes(decoded.data))

    with mcap_path.open("rb") as stream:
        for schema, _channel, message in make_reader(stream, validate_crcs=True).iter_messages(
            topics=[IMU_TOPIC]
        ):
            if schema is None or schema.name != "sensor_msgs/msg/Imu":
                raise ValueError("unexpected wrist IMU schema")
            timestamp, frame_id, acceleration, angular_velocity = _decode_imu(message.data)
            imu_timestamps.append(timestamp)
            imu_frame_ids.add(frame_id)
            accelerations.append(acceleration)
            angular_velocities.append(angular_velocity)

    if not video_timestamps or not imu_timestamps or video_formats != {"h264"}:
        raise ValueError("required video/IMU streams are empty or use an unexpected format")
    video_bytes = b"".join(video_packets)
    decoded_dimensions: set[tuple[int, int]] = set()
    decoded_frames = 0
    with av.open(io.BytesIO(video_bytes), format="h264", mode="r") as container:
        for frame in container.decode(video=0):
            decoded_frames += 1
            decoded_dimensions.add((frame.width, frame.height))
    if decoded_frames != len(video_packets) or len(decoded_dimensions) != 1:
        raise ValueError("decoded video frame count/dimensions do not match the stream")

    nearest_deltas: list[float] = []
    for timestamp in video_timestamps:
        position = bisect.bisect_left(imu_timestamps, timestamp)
        candidates = []
        if position:
            candidates.append(abs(timestamp - imu_timestamps[position - 1]))
        if position < len(imu_timestamps):
            candidates.append(abs(timestamp - imu_timestamps[position]))
        nearest_deltas.append(min(candidates) / 1_000_000)

    return {
        "format": "egoviz-wrist-video-imu-dev-smoke-v1",
        "scope": (
            "one deterministic development-group source record; "
            "decode and timestamp smoke only"
        ),
        "index": {"sha256": index_hash, "chunk_id": CHUNK_ID, "group_id": group_id},
        "split": {
            "seed": 17,
            "algorithm": SPLIT_PREFIX,
            "side": "development",
            "sealed_audit": False,
        },
        "mcap": {
            "bytes": mcap_path.stat().st_size,
            "sha256": _sha256(mcap_path),
            "crc_validation": "passed",
            "topic_counts": dict(sorted(topic_counts.items())),
        },
        "wrist_video": {
            "topic": VIDEO_TOPIC,
            "schema": "foxglove.CompressedVideo/protobuf",
            "format": "h264",
            "messages": len(video_packets),
            "decoded_frames": decoded_frames,
            "dimensions": list(next(iter(decoded_dimensions))),
            "frame_ids": sorted(video_frame_ids),
            "encoded_stream_sha256": hashlib.sha256(video_bytes).hexdigest(),
        },
        "wrist_imu": {
            "topic": IMU_TOPIC,
            "schema": "sensor_msgs/msg/Imu/cdr",
            "samples": len(imu_timestamps),
            "frame_ids": sorted(imu_frame_ids),
            "all_values_finite": True,
            "acceleration_component_ranges": [
                [
                    min(sample[axis] for sample in accelerations),
                    max(sample[axis] for sample in accelerations),
                ]
                for axis in range(3)
            ],
            "angular_velocity_component_ranges": [
                [
                    min(sample[axis] for sample in angular_velocities),
                    max(sample[axis] for sample in angular_velocities),
                ]
                for axis in range(3)
            ],
        },
        "nearest_video_to_imu_offset_ms": {
            "median": statistics.median(nearest_deltas),
            "p95": _percentile95(nearest_deltas),
            "max": max(nearest_deltas),
            "within_20ms": sum(value <= 20 for value in nearest_deltas),
        },
        "not_performed": [
            "action labels read",
            "model fitting",
            "quality evaluation",
            "sealed audit access",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--mcap", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = build_receipt(args.index, args.split, args.mcap)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(receipt, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
