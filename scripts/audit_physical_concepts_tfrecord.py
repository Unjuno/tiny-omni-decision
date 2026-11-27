"""Audit one pinned Physical Concepts freeform TFRecord shard without TensorFlow."""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import math
import struct
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO

_IMAGE_BYTES = 15 * 64 * 64 * 3
_MASK_BYTES = 15 * 64 * 64
_CRC32C_POLY = 0x82F63B78
_CRC32C_TABLE: tuple[int, ...] = ()


def _make_crc32c_table() -> tuple[int, ...]:
    values: list[int] = []
    for initial in range(256):
        value = initial
        for _ in range(8):
            value = (value >> 1) ^ (_CRC32C_POLY if value & 1 else 0)
        values.append(value)
    return tuple(values)


_CRC32C_TABLE = _make_crc32c_table()


def crc32c(data: bytes) -> int:
    value = 0xFFFFFFFF
    for byte in data:
        value = _CRC32C_TABLE[(value ^ byte) & 0xFF] ^ (value >> 8)
    return value ^ 0xFFFFFFFF


def masked_crc32c(data: bytes) -> int:
    value = crc32c(data)
    return (((value >> 15) | (value << 17)) + 0xA282EAD8) & 0xFFFFFFFF


def _read_varint(data: bytes, offset: int) -> tuple[int, int]:
    result = 0
    shift = 0
    while offset < len(data) and shift < 70:
        byte = data[offset]
        offset += 1
        result |= (byte & 0x7F) << shift
        if byte < 0x80:
            return result, offset
        shift += 7
    raise ValueError("invalid or truncated protobuf varint")


def _wire_fields(data: bytes) -> Iterator[tuple[int, int, int | bytes]]:
    offset = 0
    while offset < len(data):
        key, offset = _read_varint(data, offset)
        number, wire_type = key >> 3, key & 7
        if number == 0:
            raise ValueError("protobuf field number 0 is invalid")
        if wire_type == 0:
            value, offset = _read_varint(data, offset)
        elif wire_type == 1:
            end = offset + 8
            if end > len(data):
                raise ValueError("truncated protobuf fixed64 field")
            value = data[offset:end]
            offset = end
        elif wire_type == 2:
            length, offset = _read_varint(data, offset)
            end = offset + length
            if end > len(data):
                raise ValueError("truncated protobuf bytes field")
            value = data[offset:end]
            offset = end
        elif wire_type == 5:
            end = offset + 4
            if end > len(data):
                raise ValueError("truncated protobuf fixed32 field")
            value = data[offset:end]
            offset = end
        else:
            raise ValueError(f"unsupported protobuf wire type: {wire_type}")
        yield number, wire_type, value


def _feature_map(example: bytes) -> dict[str, tuple[str, list[bytes] | list[float]]]:
    example_fields = list(_wire_fields(example))
    features_fields = [value for number, wire, value in example_fields if number == 1 and wire == 2]
    if len(features_fields) != 1 or not isinstance(features_fields[0], bytes):
        raise ValueError("TF Example must contain exactly one Features field")
    result: dict[str, tuple[str, list[bytes] | list[float]]] = {}
    for number, wire_type, entry_value in _wire_fields(features_fields[0]):
        if number != 1 or wire_type != 2 or not isinstance(entry_value, bytes):
            continue
        key: str | None = None
        feature_payload: bytes | None = None
        for entry_number, entry_wire, entry_field in _wire_fields(entry_value):
            if entry_number == 1 and entry_wire == 2 and isinstance(entry_field, bytes):
                key = entry_field.decode("utf-8")
            elif entry_number == 2 and entry_wire == 2 and isinstance(entry_field, bytes):
                feature_payload = entry_field
        if key is None or feature_payload is None or key in result:
            raise ValueError("invalid or duplicate TF Example feature-map entry")
        feature_kind: str | None = None
        feature_values: list[bytes] | list[float] | None = None
        for feature_number, feature_wire, feature_value in _wire_fields(feature_payload):
            if feature_number == 1 and feature_wire == 2 and isinstance(feature_value, bytes):
                values = [
                    value
                    for item_number, item_wire, value in _wire_fields(feature_value)
                    if item_number == 1 and item_wire == 2 and isinstance(value, bytes)
                ]
                feature_kind, feature_values = "bytes", values
            elif feature_number == 2 and feature_wire == 2 and isinstance(feature_value, bytes):
                packed = [
                    value
                    for item_number, item_wire, value in _wire_fields(feature_value)
                    if item_number == 1 and item_wire == 2 and isinstance(value, bytes)
                ]
                values: list[float] = []
                for value in packed:
                    if len(value) % 4:
                        raise ValueError("invalid packed float-list length")
                    values.extend(struct.unpack(f"<{len(value) // 4}f", value))
                feature_kind, feature_values = "float", values
        if feature_kind is None or feature_values is None:
            raise ValueError(f"unsupported or empty TF Example feature: {key}")
        result[key] = feature_kind, feature_values
    return result


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    data = stream.read(size)
    if len(data) != size:
        raise ValueError("truncated TFRecord stream")
    return data


def iter_tfrecord_examples(path: Path) -> Iterator[bytes]:
    """Read GZIP-compressed TFRecord examples and validate TensorFlow CRC32C framing."""
    with gzip.open(path, "rb") as stream:
        while True:
            length_bytes = stream.read(8)
            if not length_bytes:
                return
            if len(length_bytes) != 8:
                raise ValueError("truncated TFRecord length")
            expected_length_crc = struct.unpack("<I", _read_exact(stream, 4))[0]
            if expected_length_crc != masked_crc32c(length_bytes):
                raise ValueError("TFRecord length CRC32C mismatch")
            length = struct.unpack("<Q", length_bytes)[0]
            if length > _IMAGE_BYTES + _MASK_BYTES + 64_000:
                raise ValueError(f"implausibly large TFRecord Example: {length} bytes")
            example = _read_exact(stream, length)
            expected_data_crc = struct.unpack("<I", _read_exact(stream, 4))[0]
            if expected_data_crc != masked_crc32c(example):
                raise ValueError("TFRecord Example CRC32C mismatch")
            yield example


def audit_shard(
    path: Path,
    *,
    object_name: str,
    generation: str,
    expected_size: int,
    expected_md5_base64: str,
) -> dict[str, object]:
    """Validate a pinned freeform shard and return data-only identity accounting."""
    path = path.resolve(strict=True)
    md5 = hashlib.md5()
    sha256 = hashlib.sha256()
    byte_count = 0
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            md5.update(chunk)
            sha256.update(chunk)
            byte_count += len(chunk)
    expected_md5 = base64.b64decode(expected_md5_base64, validate=True)
    if byte_count != expected_size or md5.digest() != expected_md5:
        raise ValueError("downloaded shard size or GCS MD5 does not match pinned metadata")

    row_hashes: list[str] = []
    image_hashes: list[str] = []
    mask_hashes: list[str] = []
    camera_pose_count = 0
    for index, serialized in enumerate(iter_tfrecord_examples(path)):
        features = _feature_map(serialized)
        if set(features) != {"image", "mask", "camera_pose"}:
            raise ValueError(f"row {index} has unexpected features: {sorted(features)}")
        image_kind, image_values = features["image"]
        mask_kind, mask_values = features["mask"]
        pose_kind, pose_values = features["camera_pose"]
        if image_kind != "bytes" or mask_kind != "bytes" or pose_kind != "float":
            raise ValueError(f"row {index} has a feature type mismatch")
        if len(image_values) != 1 or len(image_values[0]) != _IMAGE_BYTES:
            raise ValueError(f"row {index} image is not [15,64,64,3] uint8")
        if len(mask_values) != 1 or len(mask_values[0]) != _MASK_BYTES:
            raise ValueError(f"row {index} mask is not [15,64,64] uint8")
        if len(pose_values) != 90 or not all(math.isfinite(value) for value in pose_values):
            raise ValueError(f"row {index} camera_pose is not 15x6 finite float32")
        row_hashes.append(hashlib.sha256(serialized).hexdigest())
        image_hashes.append(hashlib.sha256(image_values[0]).hexdigest())
        mask_hashes.append(hashlib.sha256(mask_values[0]).hexdigest())
        camera_pose_count += 1

    if not row_hashes:
        raise ValueError("TFRecord shard contains no examples")
    return {
        "status": "complete",
        "audited_at_utc": datetime.now(UTC).isoformat(),
        "object_name": object_name,
        "generation": generation,
        "source_file": str(path),
        "source_bytes": byte_count,
        "source_md5_base64": base64.b64encode(md5.digest()).decode("ascii"),
        "source_sha256": sha256.hexdigest(),
        "records": len(row_hashes),
        "unique_records": len(set(row_hashes)),
        "duplicate_record_count": len(row_hashes) - len(set(row_hashes)),
        "unique_rgb_sequences": len(set(image_hashes)),
        "duplicate_rgb_sequence_count": len(image_hashes) - len(set(image_hashes)),
        "unique_mask_sequences": len(set(mask_hashes)),
        "camera_pose_records": camera_pose_count,
        "frame_shape": [15, 64, 64, 3],
        "grouping_unit": "whole TFRecord row; RGB sequence SHA-256 also emitted per row",
        "record_sha256": row_hashes,
        "rgb_sequence_sha256": image_hashes,
        "mask_sequence_sha256": mask_hashes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-file", type=Path, required=True)
    parser.add_argument("--object-name", required=True)
    parser.add_argument("--generation", required=True)
    parser.add_argument("--expected-size", type=int, required=True)
    parser.add_argument("--expected-md5-base64", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_shard(
        args.source_file,
        object_name=args.object_name,
        generation=args.generation,
        expected_size=args.expected_size,
        expected_md5_base64=args.expected_md5_base64,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        key: value
        for key, value in report.items()
        if key not in {"record_sha256", "rgb_sequence_sha256", "mask_sequence_sha256"}
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
