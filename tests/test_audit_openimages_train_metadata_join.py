import hashlib

import pytest

from scripts.audit_openimages_train_metadata_join import metadata_ids, stream_lines


def test_metadata_ids_require_unique_train_and_declared_license():
    row = {"ImageID": "a", "Subset": "train",
           "License": "https://creativecommons.org/licenses/by/2.0/"}
    assert metadata_ids([row]) == {"a"}
    with pytest.raises(ValueError, match="duplicate"):
        metadata_ids([row, row])
    with pytest.raises(ValueError, match="subset"):
        metadata_ids([row | {"Subset": "validation"}])
    with pytest.raises(ValueError, match="license"):
        metadata_ids([row | {"License": "unknown"}])


def test_stream_preserves_raw_hash_across_chunks():
    class Response:
        def iter_content(self, chunk_size):
            yield b"id\n"
            yield b"one\ntwo"

    state = {"sha": hashlib.sha256(), "bytes": 0, "logged": 0}
    assert list(stream_lines(Response(), state, 10)) == ["id", "one", "two"]
    assert state["sha"].hexdigest() == hashlib.sha256(b"id\none\ntwo").hexdigest()
