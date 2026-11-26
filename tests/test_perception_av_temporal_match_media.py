from __future__ import annotations

import zlib
from io import BytesIO
from zipfile import ZIP_BZIP2, ZIP_DEFLATED, ZipFile

import pytest

from scripts.acquire_perception_temporal_match_train_media import extract_member_bytes


class MemoryRange:
    def __init__(self, data: bytes) -> None:
        self.data = data

    def read_range(self, start: int, size: int) -> bytes:
        return self.data[start : start + size]


@pytest.mark.parametrize("compression", [ZIP_DEFLATED, ZIP_BZIP2])
def test_direct_zip_member_extraction_checks_size_and_crc(compression: int) -> None:
    payload = b"pinned test media bytes\0" * 4096
    buffer = BytesIO()
    with ZipFile(buffer, "w", compression=compression) as archive:
        archive.writestr("videos/video_1.mp4", payload)
    archive_bytes = buffer.getvalue()
    with ZipFile(BytesIO(archive_bytes)) as archive:
        member = archive.getinfo("videos/video_1.mp4")

    assert extract_member_bytes(MemoryRange(archive_bytes), member) == payload


def test_direct_zip_member_extraction_rejects_corrupt_compressed_bytes() -> None:
    buffer = BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("videos/video_2.mp4", b"media payload" * 500)
    archive_bytes = bytearray(buffer.getvalue())
    with ZipFile(BytesIO(archive_bytes)) as archive:
        member = archive.getinfo("videos/video_2.mp4")
    compressed_start = member.header_offset + 30 + len(member.filename.encode())
    archive_bytes[compressed_start + 3] ^= 0x10

    with pytest.raises((ValueError, RuntimeError, zlib.error)):
        extract_member_bytes(MemoryRange(bytes(archive_bytes)), member)
