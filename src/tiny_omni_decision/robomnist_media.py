"""Pinned ZIP-member helpers for bounded, in-memory RoboMNIST reads."""

from __future__ import annotations

import struct
import zlib
from collections.abc import Mapping


def compressed_member_range(local_header: bytes, member: Mapping[str, object]) -> tuple[int, int]:
    """Return the local-header tail range after checking identity and compression."""
    if len(local_header) != 30:
        raise ValueError("a complete 30-byte ZIP local header is required")
    (
        signature,
        _version,
        flags,
        method,
        _time,
        _date,
        _crc,
        _compressed_size,
        _uncompressed_size,
        filename_length,
        extra_length,
    ) = struct.unpack("<IHHHHHIIIHH", local_header)
    name = member.get("name")
    offset = member.get("offset")
    compressed_size = member.get("compressed_size")
    compression = member.get("compression")
    if (
        signature != 0x04034B50
        or flags & 1
        or not isinstance(name, str)
        or not name
        or type(offset) is not int
        or offset < 0
        or type(compressed_size) is not int
        or compressed_size < 0
        or type(compression) is not int
        or method != compression
    ):
        raise ValueError("local ZIP header does not match pinned member metadata")
    return offset + 30, filename_length + extra_length + compressed_size


def decode_pinned_member(
    local_header: bytes,
    local_tail: bytes,
    member: Mapping[str, object],
) -> bytes:
    """Decode one ranged ZIP member and enforce exact name, size and CRC."""
    start, length = compressed_member_range(local_header, member)
    del start
    name = member.get("name")
    compression = member.get("compression")
    expected_size = member.get("size")
    expected_crc = member.get("crc32")
    if (
        not isinstance(name, str)
        or type(compression) is not int
        or type(expected_size) is not int
        or expected_size < 0
        or not isinstance(expected_crc, str)
        or len(local_tail) != length
    ):
        raise ValueError("ranged ZIP member or pinned integrity metadata is incomplete")
    (
        _signature,
        _version,
        flags,
        method,
        _time,
        _date,
        _crc,
        _compressed_size,
        _uncompressed_size,
        filename_length,
        _extra_length,
    ) = struct.unpack("<IHHHHHIIIHH", local_header)
    name_bytes = name.encode("utf-8" if flags & 0x800 else "cp437")
    if method != compression or local_tail[:filename_length] != name_bytes:
        raise ValueError("ranged local filename or compression differs from pinned metadata")
    compressed = local_tail[filename_length + _extra_length :]
    if compression == 0:
        content = compressed
    elif compression == 8:
        try:
            content = zlib.decompress(compressed, -zlib.MAX_WBITS)
        except zlib.error as exc:
            raise ValueError("ranged deflate member is corrupt") from exc
    else:
        raise ValueError(f"unsupported ZIP compression method: {compression}")
    try:
        crc = int(expected_crc, 16)
    except ValueError as exc:
        raise ValueError("pinned member CRC must be hexadecimal") from exc
    if len(content) != expected_size or (zlib.crc32(content) & 0xFFFFFFFF) != crc:
        raise ValueError("expanded ZIP member size or CRC differs from pin")
    return content
