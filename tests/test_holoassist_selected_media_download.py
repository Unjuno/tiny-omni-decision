from __future__ import annotations

import hashlib

import pytest

from scripts.download_holoassist_selected_media import (
    download_member,
    validate_member_suffix,
)


class FakeResponse:
    def __init__(self, *, status: int, payload: bytes, total: int, offset: int, etag: str) -> None:
        self.status_code = status
        self.headers = {
            "Content-Range": f"bytes {offset}-{offset + len(payload) - 1}/{total}",
            "ETag": etag,
        }
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, chunk_size: int):
        del chunk_size
        yield self.payload[:3]
        yield b""
        yield self.payload[3:]

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return None


class FakeSession:
    def __init__(self, response: FakeResponse) -> None:
        self.response = response

    def get(self, *_args, **_kwargs):
        return self.response


def test_selected_media_download_checks_range_etag_size_and_hash(tmp_path) -> None:
    payload = b"selected-train-video"
    response = FakeResponse(
        status=206,
        payload=payload,
        total=500,
        offset=100,
        etag='"pinned"',
    )
    destination = tmp_path / "train" / "session.mp4"

    result = download_member(
        FakeSession(response),
        url="https://example.invalid/archive.tar",
        etag='"pinned"',
        archive_size=500,
        offset=100,
        size=len(payload),
        destination=destination,
    )

    assert destination.read_bytes() == payload
    assert result["sha256"] == hashlib.sha256(payload).hexdigest()
    assert result["size_bytes"] == len(payload)


def test_selected_media_download_fails_closed_on_wrong_etag(tmp_path) -> None:
    response = FakeResponse(status=206, payload=b"x", total=50, offset=10, etag='"wrong"')

    with pytest.raises(RuntimeError, match="failed to fetch"):
        download_member(
            FakeSession(response),
            url="https://example.invalid/archive.tar",
            etag='"pinned"',
            archive_size=50,
            offset=10,
            size=1,
            destination=tmp_path / "session.mp4",
            retries=1,
        )
    assert not (tmp_path / "session.mp4").exists()


def test_selected_media_download_refuses_overwrite(tmp_path) -> None:
    destination = tmp_path / "session.mp4"
    destination.write_bytes(b"keep")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        download_member(
            FakeSession(FakeResponse(status=206, payload=b"x", total=50, offset=0, etag='"e"')),
            url="https://example.invalid/archive.tar",
            etag='"e"',
            archive_size=50,
            offset=0,
            size=1,
            destination=destination,
        )
    assert destination.read_bytes() == b"keep"


def test_selected_media_download_resumes_only_from_verified_partial_prefix(tmp_path) -> None:
    payload = b"selected-train-video"
    destination = tmp_path / "train" / "session.mp4"
    destination.parent.mkdir(parents=True)
    destination.with_suffix(".mp4.part").write_bytes(payload[:5])
    response = FakeResponse(
        status=206,
        payload=payload[5:],
        total=500,
        offset=105,
        etag='"pinned"',
    )

    result = download_member(
        FakeSession(response),
        url="https://example.invalid/archive.tar",
        etag='"pinned"',
        archive_size=500,
        offset=100,
        size=len(payload),
        destination=destination,
        retries=1,
        resume_partial=True,
    )

    assert destination.read_bytes() == payload
    assert result["sha256"] == hashlib.sha256(payload).hexdigest()


def test_selected_media_suffix_accepts_explicit_pitchshift_companion() -> None:
    validate_member_suffix(
        "session/Export_py/Video_pitchshift.mp4",
        "/Export_py/Video_pitchshift.mp4",
    )


def test_selected_media_suffix_keeps_compressed_default_distinct() -> None:
    validate_member_suffix(
        "session/Export_py/Video_compress.mp4",
        "/Export_py/Video_compress.mp4",
    )
    with pytest.raises(ValueError, match="unexpected video member path"):
        validate_member_suffix(
            "session/Export_py/Video_pitchshift.mp4",
            "/Export_py/Video_compress.mp4",
        )
