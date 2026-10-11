from unittest.mock import Mock, patch

import pytest

from scripts.fetch_starss23_smoke import HttpRangeReader, metadata_inventory


@pytest.mark.parametrize(
    "status,header",
    [
        (200, None),
        (206, "bytes 0-3/100"),
        (206, "bytes 10-13/101"),
    ],
)
def test_range_reader_rejects_unbounded_or_wrong_response_before_read(status, header):
    reader = HttpRangeReader("https://example.invalid/archive", 100)
    reader.seek(10)
    response = Mock(status=status, headers={"Content-Range": header})
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    with patch("urllib.request.urlopen", return_value=response):
        with pytest.raises(ValueError, match="exact requested"):
            reader.read(4)
    response.read.assert_not_called()


def test_range_budget_prevents_full_archive_download():
    reader = HttpRangeReader("https://example.invalid/archive", 100, budget=4)
    with patch("urllib.request.urlopen") as open_url:
        with pytest.raises(ValueError, match="budget"):
            reader.read()
        open_url.assert_not_called()


def test_metadata_split_groups_by_room_not_clip():
    names = [
        "metadata_dev/dev-train-tau/fold3_room6_mix001.csv",
        "metadata_dev/dev-test-tau/fold4_room6_mix002.csv",
    ]
    with pytest.raises(ValueError, match="room overlaps"):
        metadata_inventory(names)
    rows = metadata_inventory([names[0], names[1].replace("room6", "room7")])
    assert {r["room"] for r in rows} == {"tau:room6", "tau:room7"}


def test_metadata_rejects_duplicate_recording_identity():
    name = "metadata_dev/dev-train-tau/fold3_room6_mix001.csv"
    with pytest.raises(ValueError, match="duplicate recording"):
        metadata_inventory([name, name])


def test_range_reader_preserves_seek_and_accounts_exact_bytes():
    reader = HttpRangeReader("https://example.invalid/archive", 100, budget=4)
    assert reader.seek(-4, 2) == 96
    response = Mock(status=206, headers={"Content-Range": "bytes 96-99/100"})
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    response.read.return_value = b"abcd"
    with patch("urllib.request.urlopen", return_value=response):
        assert reader.read(4) == b"abcd"
    assert reader.tell() == 100
    assert reader.transferred == 4
    assert reader.read() == b""


def test_range_reader_rejects_truncated_body():
    reader = HttpRangeReader("https://example.invalid/archive", 100)
    response = Mock(status=206, headers={"Content-Range": "bytes 0-3/100"})
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    response.read.return_value = b"abc"
    with patch("urllib.request.urlopen", return_value=response):
        with pytest.raises(ValueError, match="length mismatch"):
            reader.read(4)


def test_short_response_retry_keeps_range_and_counts_failed_bytes():
    reader = HttpRangeReader("https://example.invalid/archive", 100, budget=8, retries=2)
    response = Mock(status=206, headers={"Content-Range": "bytes 0-3/100"})
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    response.read.side_effect = [b"ab", b"abcd"]
    with patch("urllib.request.urlopen", return_value=response) as opened:
        assert reader.read(4) == b"abcd"
    assert reader.tell() == 4
    assert reader.transferred == 6
    assert opened.call_count == 2
    assert all(call.args[0].get_header("Range") == "bytes=0-3" for call in opened.call_args_list)


def test_short_response_retries_obey_transfer_budget():
    reader = HttpRangeReader("https://example.invalid/archive", 100, budget=4, retries=2)
    response = Mock(status=206, headers={"Content-Range": "bytes 0-3/100"})
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    response.read.return_value = b"ab"
    with patch("urllib.request.urlopen", return_value=response) as opened:
        with pytest.raises(ValueError, match="budget"):
            reader.read(4)
    assert opened.call_count == 1
    assert reader.tell() == 0
