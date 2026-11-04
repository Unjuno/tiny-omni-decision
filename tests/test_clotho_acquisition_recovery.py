import hashlib
import json
import urllib.error
from unittest.mock import Mock, patch

import pytest

from scripts.fetch_clotho_development_media import HASHES, TransientRangeReader, reusable_records


def response():
    value = Mock(status=206, headers={"Content-Range": "bytes 0-3/100"})
    value.__enter__ = Mock(return_value=value)
    value.__exit__ = Mock(return_value=False)
    value.read.return_value = b"abcd"
    return value


def test_gateway_retry_keeps_exact_range_and_position():
    reader = TransientRangeReader("https://example.invalid/a", 100, budget=4)
    error = urllib.error.HTTPError(reader.url, 504, "gateway", {}, None)
    with (
        patch("urllib.request.urlopen", side_effect=[error, response()]) as opened,
        patch("time.sleep"),
    ):
        assert reader.read(4) == b"abcd"
    assert reader.tell() == reader.transferred == 4
    assert opened.call_count == 2
    assert all(c.args[0].get_header("Range") == "bytes=0-3" for c in opened.call_args_list)


def test_gateway_retries_are_bounded_and_permission_failure_is_not_retried():
    for status, calls in [(504, 3), (403, 1)]:
        reader = TransientRangeReader("https://example.invalid/a", 100)
        error = urllib.error.HTTPError(reader.url, status, "failure", {}, None)
        with patch("urllib.request.urlopen", side_effect=error) as opened, patch("time.sleep"):
            with pytest.raises(urllib.error.HTTPError):
                reader.read(4)
        assert opened.call_count == calls
        assert reader.tell() == 0


def test_invalid_range_is_not_retried():
    bad = response()
    bad.status = 200
    reader = TransientRangeReader("https://example.invalid/a", 100)
    with patch("urllib.request.urlopen", return_value=bad) as opened:
        with pytest.raises(ValueError, match="exact requested"):
            reader.read(4)
    assert opened.call_count == 1


def test_partial_terminal_reuse_checks_bytes_and_provenance(tmp_path):
    data = b"verified payload"
    wanted = dict(source_id="1", file_name="one.wav", uploader="u", license="allow", split="train")
    row = dict(
        source_id="1",
        member="audio_files/one.wav",
        uploader="u",
        license="allow",
        split="train",
        sha256=hashlib.sha256(data).hexdigest(),
        bytes=len(data),
    )
    report = dict(status="FAIL", split_hashes=HASHES, files=[row])
    (tmp_path / "result.json").write_text(json.dumps(report))
    (tmp_path / "1.wav").write_bytes(data)
    assert reusable_records(tmp_path, {"1": wanted})["1"][1] == row["sha256"]
    (tmp_path / "1.wav").write_bytes(data + b"corrupt")
    with pytest.raises(ValueError, match="hash/length"):
        reusable_records(tmp_path, {"1": wanted})


def test_running_source_cannot_be_reused(tmp_path):
    (tmp_path / "result.json").write_text(json.dumps(dict(status="RUNNING", split_hashes=HASHES)))
    with pytest.raises(ValueError, match="terminal"):
        reusable_records(tmp_path, {})
