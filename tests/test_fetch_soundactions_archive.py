import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.fetch_soundactions_archive import _fetch_chunk, chunk_ranges, fetch_archive


def test_chunk_ranges_cover_archive_without_gaps():
    assert chunk_ranges(10, 4) == [(0, 3), (4, 7), (8, 9)]


@pytest.mark.parametrize("total,chunk", [(0, 1), (1, 0), (-1, 2)])
def test_chunk_ranges_reject_nonpositive_sizes(total, chunk):
    with pytest.raises(ValueError, match="positive"):
        chunk_ranges(total, chunk)


def test_fetch_refuses_to_overwrite_existing_archive(tmp_path: Path):
    output = tmp_path / "already-there.zip"
    output.write_bytes(b"preserve")
    with pytest.raises(FileExistsError, match="overwrite"):
        fetch_archive(
            url="https://example.invalid/archive.zip",
            expected_bytes=7,
            expected_md5="00000000000000000000000000000000",
            output=output,
        )
    assert output.read_bytes() == b"preserve"


def _fake_range_response(monkeypatch, data: bytes, *, fail_first: bool = False):
    calls = []

    def run(command, **_kwargs):
        range_value = command[command.index("--range") + 1]
        start, end = map(int, range_value.split("-"))
        header_path = Path(command[command.index("--dump-header") + 1])
        output_path = Path(command[command.index("--output") + 1])
        calls.append((start, end))
        header_path.write_text(
            f"HTTP/1.1 206 PARTIAL_CONTENT\r\n"
            f"Content-Range: bytes {start}-{end}/{len(data)}\r\n\r\n",
            encoding="ascii",
        )
        body = data[start : end + 1]
        if fail_first and len(calls) == 1:
            output_path.write_bytes(body[:3])
            return SimpleNamespace(returncode=28)
        output_path.write_bytes(body)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("scripts.fetch_soundactions_archive.subprocess.run", run)
    monkeypatch.setattr("scripts.fetch_soundactions_archive.time.sleep", lambda _seconds: None)
    return calls


def test_fetch_chunk_resumes_only_from_digest_verified_partial(tmp_path, monkeypatch):
    chunks = tmp_path / "chunks"
    chunks.mkdir()
    data = b"0123456789abcdefghij"
    partial = chunks / "0000.part"
    prefix = data[10:13]
    partial.write_bytes(prefix)
    partial.with_suffix(".part.sha256").write_text(
        hashlib.sha256(prefix).hexdigest() + "\n", encoding="ascii"
    )
    (chunks / "0000.headers").write_text(
        "HTTP/1.1 206 PARTIAL_CONTENT\r\nContent-Range: bytes 10-19/20\r\n",
        encoding="ascii",
    )
    calls = _fake_range_response(monkeypatch, data)

    result = _fetch_chunk(
        curl="curl",
        url="https://example.invalid/data",
        total_bytes=20,
        index=0,
        start=10,
        end=19,
        chunks_dir=chunks,
    )

    assert calls == [(13, 19)]
    assert result.read_bytes() == data[10:20]


def test_fetch_chunk_quarantines_partial_without_matching_digest(tmp_path, monkeypatch):
    chunks = tmp_path / "chunks"
    chunks.mkdir()
    data = b"0123456789abcdefghij"
    partial = chunks / "0000.part"
    partial.write_bytes(b"wrong")
    partial.with_suffix(".part.sha256").write_text("stale\n", encoding="ascii")
    (chunks / "0000.headers").write_text(
        "HTTP/1.1 206 PARTIAL_CONTENT\r\nContent-Range: bytes 10-19/20\r\n",
        encoding="ascii",
    )
    calls = _fake_range_response(monkeypatch, data)

    result = _fetch_chunk(
        curl="curl",
        url="https://example.invalid/data",
        total_bytes=20,
        index=0,
        start=10,
        end=19,
        chunks_dir=chunks,
    )

    assert calls == [(10, 19)]
    assert result.read_bytes() == data[10:20]
    assert (chunks / "0000.part.untrusted").read_bytes() == b"wrong"


def test_fetch_chunk_discards_incomplete_http_body_before_retry(tmp_path, monkeypatch):
    chunks = tmp_path / "chunks"
    chunks.mkdir()
    data = b"0123456789abcdefghij"
    calls = _fake_range_response(monkeypatch, data, fail_first=True)

    result = _fetch_chunk(
        curl="curl",
        url="https://example.invalid/data",
        total_bytes=20,
        index=0,
        start=10,
        end=19,
        chunks_dir=chunks,
    )

    assert calls == [(10, 19), (10, 19)]
    assert result.read_bytes() == data[10:20]
