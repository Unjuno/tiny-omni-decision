from __future__ import annotations

import io
import tarfile

import pytest

from scripts.index_holoassist_range_tar import HttpRangeReader, index_tar, safe_member_path


class FakeResponse:
    def __init__(self, start: int, end: int, payload: bytes, total: int, etag: str) -> None:
        self.status_code = 206
        self.headers = {"Content-Range": f"bytes {start}-{end}/{total}", "ETag": etag}
        self.content = payload

    def raise_for_status(self) -> None:
        return None


class FakeSession:
    def __init__(self, payload: bytes, etag: str = '"fixture"') -> None:
        self.payload = payload
        self.etag = etag

    def get(self, _url, *, headers, timeout):
        del timeout
        start, end = map(int, headers["Range"].removeprefix("bytes=").split("-"))
        return FakeResponse(
            start, end, self.payload[start : end + 1], len(self.payload), self.etag
        )


def make_tar() -> bytes:
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w:") as archive:
        member = tarfile.TarInfo("recording/Video_compress.mp4")
        content = b"media-payload" * 100
        member.size = len(content)
        archive.addfile(member, io.BytesIO(content))
        directory = tarfile.TarInfo("recording/Export_py/Video/")
        directory.type = tarfile.DIRTYPE
        archive.addfile(directory)
    return payload.getvalue()


def test_range_reader_seeks_reads_and_reuses_verified_cache() -> None:
    source = bytes(range(64))
    session = FakeSession(source)
    reader = HttpRangeReader(
        "https://example.invalid/archive.tar",
        len(source),
        '"fixture"',
        block_size=16,
        session=session,
    )

    assert reader.read(4) == source[:4]
    assert reader.seek(2) == 2
    assert reader.read(5) == source[2:7]
    assert reader.range_requests == 1
    assert reader.range_bytes == 16


def test_range_reader_fails_closed_on_etag_change() -> None:
    session = FakeSession(bytes(range(64)), etag='"changed"')
    reader = HttpRangeReader(
        "https://example.invalid/archive.tar",
        64,
        '"fixture"',
        block_size=16,
        retries=1,
        session=session,
    )

    with pytest.raises(RuntimeError, match="verified range"):
        reader.read(1)


def test_tar_index_reads_headers_without_downloading_member_payload() -> None:
    source = make_tar()
    reader = HttpRangeReader(
        "https://example.invalid/archive.tar",
        len(source),
        '"fixture"',
        block_size=512,
        session=FakeSession(source),
    )

    result = index_tar(reader, {"recording"}, progress_every=0)

    assert result["scanned_member_count"] == 2
    assert result["allowed_member_count"] == 2
    assert result["video_members"][0]["name"] == "recording/Video_compress.mp4"
    assert result["audio_members"] == []
    assert reader.range_bytes < len(source)


def test_tar_index_does_not_persist_non_allowed_session_members() -> None:
    source = make_tar()
    reader = HttpRangeReader(
        "https://example.invalid/archive.tar",
        len(source),
        '"fixture"',
        block_size=512,
        session=FakeSession(source),
    )

    result = index_tar(reader, {"another-recording"}, progress_every=0)

    assert result["scanned_member_count"] == 2
    assert result["allowed_member_count"] == 0
    assert result["video_members"] == []


@pytest.mark.parametrize("name", ["../outside.mp4", "/absolute.mp4", "nested\\path.mp4"])
def test_archive_path_guard_rejects_unsafe_member_names(name: str) -> None:
    assert not safe_member_path(name)


def test_archive_path_guard_accepts_relative_posix_members() -> None:
    assert safe_member_path("recording/Video_compress.mp4")
