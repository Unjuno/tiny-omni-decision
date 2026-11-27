from __future__ import annotations

import zipfile

import pytest

from scripts.fetch_bsd10k_allowed_audio import (
    _extract_member_bytes,
    _normalize_ranges,
    _parse_multipart_ranges,
)


def test_member_extraction_requests_only_member_header_and_payload(tmp_path) -> None:
    archive_path = tmp_path / "audio.zip"
    payload = b"RIFF" + bytes(range(256)) * 100
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("audio/123.wav", payload)
        archive.writestr("audio/456.wav", b"restricted-neighbor")
    archive_bytes = archive_path.read_bytes()
    with zipfile.ZipFile(archive_path) as archive:
        info = archive.getinfo("audio/123.wav")

    requested: list[tuple[int, int]] = []

    def read_range(offset: int, length: int) -> bytes:
        requested.append((offset, length))
        return archive_bytes[offset : offset + length]

    assert _extract_member_bytes(info, read_range) == payload
    assert len(requested) == 2
    assert requested[0][0] == info.header_offset
    assert requested[1][1] == info.compress_size


def test_multipart_parser_verifies_every_exact_range() -> None:
    body = (
        b"\n--EOSMULTIPARTBOUNDARY\nContent-Type: application/octet-stream\n"
        b"Content-Range: bytes 5-6/100\n\nAB"
        b"\n--EOSMULTIPARTBOUNDARY\nContent-Type: application/octet-stream\n"
        b"Content-Range: bytes 20-22/100\n\nXYZ"
        b"\n--EOSMULTIPARTBOUNDARY--\n"
    )
    assert _parse_multipart_ranges(body, [(5, 6), (20, 22)], 100) == {
        (5, 6): b"AB",
        (20, 22): b"XYZ",
    }
    with pytest.raises(OSError, match="differ from the exact request"):
        _parse_multipart_ranges(body, [(5, 6)], 100)


def test_range_normalization_only_merges_adjacent_ranges() -> None:
    assert _normalize_ranges([(10, 14), (0, 4), (5, 9), (20, 22)]) == [
        (0, 14),
        (20, 22),
    ]
    with pytest.raises(ValueError, match="overlapping"):
        _normalize_ranges([(0, 4), (4, 8)])
