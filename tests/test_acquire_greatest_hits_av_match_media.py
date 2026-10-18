import struct
import zlib

import pytest

from scripts.acquire_greatest_hits_av_match_media import _safe_member_bytes


def _zip_member(payload: bytes):
    name = b"vis-data-256/example_denoised.wav"
    compressor = zlib.compressobj(wbits=-15)
    compressed = compressor.compress(payload) + compressor.flush()
    crc = zlib.crc32(payload)
    header = struct.pack(
        "<4s5H3I2H",
        b"PK\x03\x04",
        20,
        0x800,
        8,
        0,
        0,
        crc,
        len(compressed),
        len(payload),
        len(name),
        0,
    )
    member = header + name + compressed
    entry = {
        "name": name.decode(),
        "bytes": len(payload),
        "compressed": len(compressed),
        "offset": 0,
        "compression": 8,
        "crc32": f"{crc:08x}",
    }
    return entry, member


def test_safe_member_reader_checks_header_size_and_crc():
    payload = b"fixed audio bytes" * 10
    entry, member = _zip_member(payload)

    actual = _safe_member_bytes(entry, lambda start, count: member[start : start + count])

    assert actual == payload


def test_safe_member_reader_rejects_crc_mismatch():
    entry, member = _zip_member(b"fixed media")
    entry["crc32"] = "00000000"

    with pytest.raises(ValueError, match="CRC"):
        _safe_member_bytes(entry, lambda start, count: member[start : start + count])


def test_safe_member_reader_rejects_expanded_payload_over_cap():
    entry, member = _zip_member(b"small")
    entry["bytes"] = 41 * 1024 * 1024

    with pytest.raises(ValueError, match="oversized"):
        _safe_member_bytes(entry, lambda start, count: member[start : start + count])
