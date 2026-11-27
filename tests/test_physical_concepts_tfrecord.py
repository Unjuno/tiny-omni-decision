from __future__ import annotations

import base64
import gzip
import hashlib
import struct

import pytest

from scripts.audit_physical_concepts_tfrecord import (
    audit_shard,
    crc32c,
    iter_tfrecord_examples,
    masked_crc32c,
)


def _varint(value: int) -> bytes:
    result = bytearray()
    while value > 0x7F:
        result.append((value & 0x7F) | 0x80)
        value >>= 7
    result.append(value)
    return bytes(result)


def _field(number: int, wire_type: int, value: bytes) -> bytes:
    key = _varint(number << 3 | wire_type)
    return key + (_varint(len(value)) + value if wire_type == 2 else value)


def _bytes_feature(value: bytes) -> bytes:
    byte_value = _field(1, 2, value)
    return _field(1, 2, byte_value)


def _float_feature(values: list[float]) -> bytes:
    packed = struct.pack(f"<{len(values)}f", *values)
    float_list = _field(1, 2, packed)
    return _field(2, 2, float_list)


def _map_entry(key: str, feature: bytes) -> bytes:
    return _field(1, 2, key.encode()) + _field(2, 2, feature)


def _example() -> bytes:
    image = bytes(range(256)) * (15 * 64 * 64 * 3 // 256)
    mask = bytes(15 * 64 * 64)
    features = b"".join(
        _field(1, 2, _map_entry(key, feature))
        for key, feature in (
            ("image", _bytes_feature(image)),
            ("mask", _bytes_feature(mask)),
            ("camera_pose", _float_feature([0.0] * 90)),
        )
    )
    return _field(1, 2, features)


def _write_tfrecord(path, examples: list[bytes]) -> None:
    body = bytearray()
    for example in examples:
        length = struct.pack("<Q", len(example))
        body.extend(length)
        body.extend(struct.pack("<I", masked_crc32c(length)))
        body.extend(example)
        body.extend(struct.pack("<I", masked_crc32c(example)))
    with gzip.open(path, "wb") as stream:
        stream.write(body)


def test_crc32c_matches_standard_check_value() -> None:
    assert crc32c(b"123456789") == 0xE3069283


def test_iter_tfrecord_examples_rejects_bad_crc(tmp_path) -> None:
    path = tmp_path / "bad.tfrecord.gz"
    length = struct.pack("<Q", 1)
    with gzip.open(path, "wb") as stream:
        stream.write(length + struct.pack("<I", 0) + b"x" + struct.pack("<I", 0))

    with pytest.raises(ValueError, match="length CRC32C"):
        list(iter_tfrecord_examples(path))


def test_audit_shard_checks_gcs_receipt_and_dataset_schema(tmp_path) -> None:
    path = tmp_path / "sample.tfrecord.gz"
    example = _example()
    _write_tfrecord(path, [example, example])
    content = path.read_bytes()

    report = audit_shard(
        path,
        object_name="freeform/train/data.tfrecord-00000-of-00100",
        generation="1599869980291930",
        expected_size=len(content),
        expected_md5_base64=base64.b64encode(hashlib.md5(content).digest()).decode(),
    )

    assert report["records"] == 2
    assert report["unique_records"] == 1
    assert report["duplicate_record_count"] == 1
    assert report["unique_rgb_sequences"] == 1
    assert report["duplicate_rgb_sequence_count"] == 1
    assert report["camera_pose_records"] == 2
    assert report["frame_shape"] == [15, 64, 64, 3]


def test_audit_shard_fails_closed_on_wrong_gcs_hash(tmp_path) -> None:
    path = tmp_path / "sample.tfrecord.gz"
    _write_tfrecord(path, [_example()])

    with pytest.raises(ValueError, match="size or GCS MD5"):
        audit_shard(
            path,
            object_name="freeform/train/data.tfrecord-00000-of-00100",
            generation="159986998998",
            expected_size=path.stat().st_size,
            expected_md5_base64=base64.b64encode(bytes(16)).decode(),
        )
