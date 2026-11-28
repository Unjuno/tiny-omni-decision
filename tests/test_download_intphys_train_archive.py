import hashlib
import urllib.error

import pytest

from scripts.download_intphys_train_archive import (
    ArchiveIntegrityError,
    download_archive,
)


class FakeResponse:
    def __init__(self, status, headers, reads):
        self.status = status
        self.headers = headers
        self._reads = list(reads)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, size):
        item = self._reads.pop(0)
        if isinstance(item, Exception):
            raise item
        if len(item) > size:
            self._reads.insert(0, item[size:])
            item = item[:size]
        return item


def test_interrupted_transfer_resumes_at_exact_partial_offset(tmp_path):
    payload = b"0123456789"
    etag = '"version-1"'
    requests = []

    def open_response(request, timeout):
        requests.append((request, timeout))
        if len(requests) == 1:
            return FakeResponse(
                200,
                {"ETag": etag, "Content-Length": str(len(payload))},
                [payload[:4], urllib.error.URLError("connection reset")],
            )
        assert request.get_header("Range") == "bytes=4-9"
        return FakeResponse(
            206,
            {
                "ETag": etag,
                "Content-Range": "bytes 4-9/10",
                "Content-Length": "6",
            },
            [payload[4:], b""],
        )

    output = tmp_path / "train.2.tar.gz.partial"
    result = download_archive(
        url="https://example.invalid/train.2.tar.gz",
        output_path=output,
        expected_bytes=len(payload),
        expected_md5=hashlib.md5(payload, usedforsecurity=False).hexdigest(),
        etag=etag,
        reserve_bytes=0,
        chunk_bytes=4,
        opener=open_response,
        sleep=lambda _seconds: None,
    )

    assert output.read_bytes() == payload
    assert result.resumed_from_bytes == 0
    assert result.http_responses == 2
    assert requests[0][0].get_header("If-match") == etag
    assert requests[1][0].get_header("If-match") == etag
    assert result.sha256 == hashlib.sha256(payload).hexdigest()


@pytest.mark.parametrize(
    "response",
    [
        FakeResponse(200, {"ETag": '"version-1"', "Content-Length": "10"}, [b"0123456789"]),
        FakeResponse(
            206,
            {
                "ETag": '"version-2"',
                "Content-Range": "bytes 3-9/10",
                "Content-Length": "7",
            },
            [b"3456789"],
        ),
        FakeResponse(
            206,
            {
                "ETag": '"version-1"',
                "Content-Range": "bytes 0-9/10",
                "Content-Length": "10",
            },
            [b"0123456789"],
        ),
    ],
)
def test_invalid_resume_response_does_not_modify_partial(tmp_path, response):
    output = tmp_path / "train.2.tar.gz.partial"
    output.write_bytes(b"abc")

    with pytest.raises(ArchiveIntegrityError):
        download_archive(
            url="https://example.invalid/train.2.tar.gz",
            output_path=output,
            expected_bytes=10,
            expected_md5="00000000000000000000000000000000",
            etag='"version-1"',
            reserve_bytes=0,
            opener=lambda *_args, **_kwargs: response,
            sleep=lambda _seconds: None,
        )

    assert output.read_bytes() == b"abc"
