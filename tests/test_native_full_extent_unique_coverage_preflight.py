import hashlib

import pytest

from scripts.native_full_extent import parent_probe
from scripts.prepare_native_full_extent_unique_coverage import serialize, serialized_sha, write


def test_preflight_writer_bytes_match_precomputed_hash(tmp_path):
    payload = {"sample": [1, 2], "name": "fresh"}
    target = tmp_path / "receipt.json"

    write(target, payload)

    assert target.read_bytes() == serialize(payload).encode("utf-8")
    assert serialized_sha(payload) == hashlib.sha256(target.read_bytes()).hexdigest()


def test_parent_probe_rejects_non_train_rows_and_duplicate_ids():
    with pytest.raises(ValueError, match="unique TRAIN rows"):
        parent_probe([{"id": "a", "split": "validation"}], [0])
    with pytest.raises(ValueError, match="unique TRAIN rows"):
        parent_probe(
            [{"id": "a", "split": "train"}, {"id": "a", "split": "train"}], [0, 1]
        )
