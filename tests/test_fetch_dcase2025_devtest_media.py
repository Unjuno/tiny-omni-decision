from __future__ import annotations

import hashlib
import io
import zipfile
import zlib

import pytest

from scripts import fetch_dcase2025_devtest_media as fetch


def _archive_member() -> tuple[bytes, zipfile.ZipInfo, bytes]:
    payload = (b"held-out media fixture" * 4096) + bytes(range(256))
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("fold3_room23_mix008.wav", payload)
        info = archive.getinfo("fold3_room23_mix008.wav")
    return buffer.getvalue(), info, payload


def test_member_fetch_streaming_checks_zip_crc_and_sha(tmp_path, monkeypatch):
    archive_bytes, info, payload = _archive_member()

    def local_range(_url, start, length, *, total_size, timeout=120):
        assert total_size == fetch.ARCHIVES["foa_dev.zip"][0]
        return archive_bytes[start : start + length]

    def local_stream(_url, start, length, destination, *, total_size, archive_name, counters):
        assert total_size == fetch.ARCHIVES["foa_dev.zip"][0]
        data = archive_bytes[start : start + length]
        destination.write_bytes(data)
        counters[archive_name] += len(data)

    monkeypatch.setattr(fetch, "request_range", local_range)
    monkeypatch.setattr(fetch, "stream_range_to_file", local_stream)
    counters = {"foa_dev.zip": 0, "video_dev.zip": 0}
    path = tmp_path / "heldout.wav"
    record = fetch.extract_member("foa_dev.zip", info, path, counters=counters)
    assert path.read_bytes() == payload
    assert record["bytes"] == len(payload)
    assert record["crc32"] == info.CRC
    assert record["sha256"] == hashlib.sha256(payload).hexdigest()
    assert not path.with_suffix(".wav.compressed.part").exists()


def test_member_fetch_rejects_corrupt_payload_and_removes_partial_output(tmp_path, monkeypatch):
    archive_bytes, info, _payload = _archive_member()

    def corrupt_range(_url, start, length, *, total_size, timeout=120):
        assert total_size == fetch.ARCHIVES["foa_dev.zip"][0]
        return archive_bytes[start : start + length]

    def corrupt_stream(_url, start, length, destination, *, total_size, archive_name, counters):
        assert total_size == fetch.ARCHIVES["foa_dev.zip"][0]
        data = bytearray(archive_bytes[start : start + length])
        data[len(data) // 2] ^= 0x80
        destination.write_bytes(data)
        counters[archive_name] += len(data)

    monkeypatch.setattr(fetch, "request_range", corrupt_range)
    monkeypatch.setattr(fetch, "stream_range_to_file", corrupt_stream)
    counters = {"foa_dev.zip": 0, "video_dev.zip": 0}
    path = tmp_path / "heldout.wav"
    with pytest.raises((ValueError, zlib.error)):
        fetch.extract_member("foa_dev.zip", info, path, counters=counters)
    assert not path.exists()


def test_stream_range_retries_truncated_http_response(tmp_path, monkeypatch):
    payload = b"compressed bytes for an exact range"
    calls = 0

    class Response:
        status = 206

        def __init__(self, data):
            self.headers = {"Content-Range": f"bytes 10-44/{fetch.ARCHIVES['foa_dev.zip'][0]}"}
            self.data = data

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self, size):
            data, self.data = self.data[:size], self.data[size:]
            return data

    def fake_urlopen(_request, timeout):
        nonlocal calls
        assert timeout == 120
        calls += 1
        return Response(payload[:7] if calls == 1 else payload)

    monkeypatch.setattr(fetch.urllib.request, "urlopen", fake_urlopen)
    destination = tmp_path / "compressed.part"
    counters = {"foa_dev.zip": 0}
    fetch.stream_range_to_file(
        "https://example.test/archive",
        10,
        len(payload),
        destination,
        total_size=fetch.ARCHIVES["foa_dev.zip"][0],
        archive_name="foa_dev.zip",
        counters=counters,
    )
    assert calls == 2
    assert destination.read_bytes() == payload
    assert counters["foa_dev.zip"] == len(payload) + 7
