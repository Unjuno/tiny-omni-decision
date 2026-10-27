import gzip
import io

import pytest

from scripts.scan_inat_photo_metadata import DigestReader, select_photo_metadata


def test_selects_all_occurrences_and_does_not_infer_unique_photo():
    content = (
        "photo_uuid\tphoto_id\tobservation_uuid\tobserver_id\tlicense\n"
        "a\t10\tu1\t7\tCC-BY\n"
        "b\t11\tu2\t8\tCC0\n"
        "c\t10\tu3\t7\tCC-BY\n"
    )
    result, count = select_photo_metadata(io.StringIO(content), {10})
    assert count == 3 and len(result[10]) == 2
    assert {item["observation_uuid"] for item in result[10]} == {"u1", "u3"}
    assert 11 not in result


def test_missing_required_column_fails_closed():
    with pytest.raises(ValueError, match="column"):
        select_photo_metadata(io.StringIO("photo_id\tlicense\n10\tCC-BY\n"), {10})


def test_gzip_reader_hashes_compressed_bytes(tmp_path):
    class Raw:
        def __init__(self, content):
            self.file = io.BytesIO(content)

        def read(self, size=-1, decode_content=False):
            assert not decode_content
            return self.file.read(size)

    payload = gzip.compress(b"test bytes")
    reader = DigestReader(Raw(payload), tmp_path)
    with gzip.GzipFile(fileobj=reader) as source:
        assert source.read() == b"test bytes"
    import hashlib

    assert reader.count == len(payload)
    assert reader.digest.hexdigest() == hashlib.sha256(payload).hexdigest()
