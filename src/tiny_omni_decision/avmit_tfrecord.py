"""Minimal, checksum-validating AVMIT SequenceExample reader without TensorFlow."""

from __future__ import annotations

import struct
from collections.abc import Iterator
from typing import BinaryIO

_CRC32C_POLYNOMIAL = 0x82F63B78
_CRC32C_MASK_DELTA = 0xA282EAD8


def _make_crc32c_table() -> tuple[int, ...]:
    table = []
    for index in range(256):
        value = index
        for _ in range(8):
            value = (value >> 1) ^ (_CRC32C_POLYNOMIAL if value & 1 else 0)
        table.append(value)
    return tuple(table)


_CRC32C_TABLE = _make_crc32c_table()


def crc32c(data: bytes) -> int:
    """Return Castagnoli CRC-32C, used by the TFRecord framing format."""
    value = 0xFFFFFFFF
    for byte in data:
        value = _CRC32C_TABLE[(value ^ byte) & 0xFF] ^ (value >> 8)
    return value ^ 0xFFFFFFFF


def _masked_crc32c(data: bytes) -> int:
    value = crc32c(data)
    return (((value >> 15) | (value << 17)) + _CRC32C_MASK_DELTA) & 0xFFFFFFFF


def iter_tfrecord_payloads(
    stream: BinaryIO, *, max_record_bytes: int = 64 * 1024 * 1024
) -> Iterator[bytes]:
    """Yield uncompressed TFRecord payloads only after validating both CRCs."""
    while True:
        length_bytes = stream.read(8)
        if not length_bytes:
            return
        if len(length_bytes) != 8:
            raise ValueError("truncated TFRecord length")
        length_crc_bytes = stream.read(4)
        if len(length_crc_bytes) != 4:
            raise ValueError("truncated TFRecord length checksum")
        expected_length_crc = struct.unpack("<I", length_crc_bytes)[0]
        if expected_length_crc != _masked_crc32c(length_bytes):
            raise ValueError("TFRecord length checksum mismatch")
        length = struct.unpack("<Q", length_bytes)[0]
        if length > max_record_bytes:
            raise ValueError(f"TFRecord payload exceeds limit: {length}")
        payload = stream.read(length)
        if len(payload) != length:
            raise ValueError("truncated TFRecord payload")
        payload_crc_bytes = stream.read(4)
        if len(payload_crc_bytes) != 4:
            raise ValueError("truncated TFRecord payload checksum")
        expected_payload_crc = struct.unpack("<I", payload_crc_bytes)[0]
        if expected_payload_crc != _masked_crc32c(payload):
            raise ValueError("TFRecord payload checksum mismatch")
        yield payload


def _read_varint(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while offset < len(data) and shift < 70:
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, offset
        shift += 7
    raise ValueError("invalid or truncated protobuf varint")


def _protobuf_fields(data: bytes) -> Iterator[tuple[int, int, bytes | int]]:
    offset = 0
    while offset < len(data):
        tag, offset = _read_varint(data, offset)
        number, wire_type = tag >> 3, tag & 7
        if number == 0:
            raise ValueError("protobuf field number zero is invalid")
        if wire_type == 0:
            value, offset = _read_varint(data, offset)
        elif wire_type == 1:
            end = offset + 8
            if end > len(data):
                raise ValueError("truncated protobuf fixed64 field")
            value, offset = data[offset:end], end
        elif wire_type == 2:
            length, offset = _read_varint(data, offset)
            end = offset + length
            if end > len(data):
                raise ValueError("truncated protobuf length-delimited field")
            value, offset = data[offset:end], end
        elif wire_type == 5:
            end = offset + 4
            if end > len(data):
                raise ValueError("truncated protobuf fixed32 field")
            value, offset = data[offset:end], end
        else:
            raise ValueError(f"unsupported protobuf wire type: {wire_type}")
        yield number, wire_type, value


def _map_entries(data: bytes) -> dict[str, bytes]:
    entries: dict[str, bytes] = {}
    for number, wire_type, entry_bytes in _protobuf_fields(data):
        if number != 1 or wire_type != 2 or not isinstance(entry_bytes, bytes):
            continue
        key = None
        value = None
        for entry_number, entry_wire, entry_value in _protobuf_fields(entry_bytes):
            if entry_number == 1 and entry_wire == 2 and isinstance(entry_value, bytes):
                key = entry_value.decode("utf-8")
            elif entry_number == 2 and entry_wire == 2 and isinstance(entry_value, bytes):
                value = entry_value
        if key is None or value is None or key in entries:
            raise ValueError("invalid or duplicate protobuf map entry")
        entries[key] = value
    return entries


def _feature_bytes(data: bytes) -> tuple[bytes, ...]:
    bytes_list = None
    for number, wire_type, value in _protobuf_fields(data):
        if number == 1 and wire_type == 2 and isinstance(value, bytes):
            if bytes_list is not None:
                raise ValueError("multiple Feature oneof values")
            bytes_list = value
    if bytes_list is None:
        raise ValueError("expected a byte-valued SequenceExample feature")
    return tuple(
        value
        for number, wire_type, value in _protobuf_fields(bytes_list)
        if number == 1 and wire_type == 2 and isinstance(value, bytes)
    )


def parse_avmit_sequence_example(payload: bytes) -> dict[str, object]:
    """Read AVMIT's filename and audio/visual byte sequences; ignore target labels."""
    context = None
    feature_lists = None
    for number, wire_type, value in _protobuf_fields(payload):
        if number == 1 and wire_type == 2 and isinstance(value, bytes):
            if context is not None:
                raise ValueError("duplicate SequenceExample context")
            context = value
        elif number == 2 and wire_type == 2 and isinstance(value, bytes):
            if feature_lists is not None:
                raise ValueError("duplicate SequenceExample feature_lists")
            feature_lists = value
    if context is None or feature_lists is None:
        raise ValueError("SequenceExample is missing context or feature_lists")

    context_features = _map_entries(context)
    filename_values = _feature_bytes(context_features["filename"])
    if len(filename_values) != 1:
        raise ValueError("SequenceExample filename must be a single byte value")
    filename = filename_values[0].decode("utf-8")

    parsed_sequences: dict[str, list[bytes]] = {}
    for key, feature_list in _map_entries(feature_lists).items():
        if key not in {"audio", "visual_object"}:
            continue
        steps = []
        for number, wire_type, feature in _protobuf_fields(feature_list):
            if number != 1 or wire_type != 2 or not isinstance(feature, bytes):
                continue
            values = _feature_bytes(feature)
            if len(values) != 1:
                raise ValueError(f"{key} timestep must have one serialized byte value")
            steps.append(values[0])
        parsed_sequences[key] = steps

    if set(parsed_sequences) != {"audio", "visual_object"}:
        raise ValueError("SequenceExample must contain audio and visual_object sequences")
    if not parsed_sequences["audio"] or not parsed_sequences["visual_object"]:
        raise ValueError("AVMIT feature sequences must be nonempty")
    return {"filename": filename, **parsed_sequences}


def decode_float32_steps(steps: list[bytes]):
    """Decode per-timestep little-endian float32 vectors with fixed finite width."""
    import numpy as np

    arrays = []
    width = None
    for index, step in enumerate(steps):
        if len(step) % 4:
            raise ValueError(f"float32 feature step {index} has an invalid byte length")
        values = np.frombuffer(step, dtype="<f4")
        if width is None:
            width = len(values)
        if not len(values) or len(values) != width:
            raise ValueError("feature vectors must have one fixed positive width")
        if not np.isfinite(values).all():
            raise ValueError("feature vectors must contain finite values")
        arrays.append(values)
    return np.stack(arrays)
