import pytest

from scripts.eidon_range_io import HTTPRangeReader


class _Response:
    def __init__(self, *, status_code, content=b"", headers=None, url=None):
        self.status_code = status_code
        self.content = content
        self.headers = headers or {}
        self.url = url or "https://cdn.example/data.bin"

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _Session:
    def __init__(self, data, *, range_status=206):
        self.data = data
        self.range_status = range_status
        self.get_calls = []

    def head(self, url, *, allow_redirects, timeout):
        assert allow_redirects is True
        return _Response(
            status_code=200,
            headers={"content-length": str(len(self.data)), "accept-ranges": "bytes"},
        )

    def get(self, url, *, headers, timeout):
        start, end = map(int, headers["Range"].removeprefix("bytes=").split("-"))
        self.get_calls.append((start, end))
        body = self.data[start : end + 1]
        return _Response(
            status_code=self.range_status,
            content=body,
            headers={"content-range": f"bytes {start}-{end}/{len(self.data)}"},
        )

    def close(self):
        pass


def test_range_reader_coalesces_reads_and_reuses_cached_blocks():
    session = _Session(b"abcdefghij")
    reader = HTTPRangeReader(
        "https://example.test/data.bin",
        max_bytes=8,
        block_size=4,
        max_cached_blocks=2,
        session=session,
    )

    assert reader.read(3) == b"abc"
    assert reader.read(3) == b"def"
    reader.seek(0)
    assert reader.read(2) == b"ab"

    assert session.get_calls == [(0, 3), (4, 7)]
    assert reader.bytes_read == 8
    assert reader.range_requests == 2


def test_range_reader_fails_before_exceeding_transfer_budget():
    session = _Session(b"abcdefghij")
    reader = HTTPRangeReader(
        "https://example.test/data.bin", max_bytes=3, block_size=4, session=session
    )

    with pytest.raises(ValueError, match="range byte budget exceeded"):
        reader.read(1)

    assert session.get_calls == []


def test_range_reader_rejects_servers_that_ignore_range_requests():
    session = _Session(b"abcdefghij", range_status=200)
    reader = HTTPRangeReader(
        "https://example.test/data.bin", max_bytes=8, block_size=4, session=session
    )

    with pytest.raises(ValueError, match="server did not honor byte range"):
        reader.read(1)
