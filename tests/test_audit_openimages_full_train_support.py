import hashlib

import pytest

from scripts.audit_openimages_full_train_support import count_support, stream_lines


def test_support_counts_unique_images_and_multiple_targets():
    groups = [
        {"positive_labels": {"a", "b"}, "negative_labels": {"c", "d", "e"}},
        {"positive_labels": {"c"}, "negative_labels": {"a", "b"}},
    ]
    images, unique, pairs, counts, availability = count_support(groups, set("abcde"))
    assert (images, unique, pairs) == (2, 1, 2)
    assert counts["a"]["eligible_target_images"] == 1
    assert counts["c"]["eligible_target_images"] == 0
    assert availability[("b", "e")] == 1


def test_support_rejects_conflicting_labels():
    with pytest.raises(ValueError, match="contradictory"):
        count_support([{"positive_labels": {"a"}, "negative_labels": {"a"}}], {"a"})


def test_stream_handles_split_lines_and_hashes_raw_bytes():
    class Response:
        def iter_content(self, chunk_size):
            yield b"a,b\n1"
            yield b",2\n3,4"

    state = {"sha": hashlib.sha256(), "bytes": 0, "logged": 0}
    assert list(stream_lines(Response(), state)) == ["a,b", "1,2", "3,4"]
    assert state["sha"].hexdigest() == hashlib.sha256(b"a,b\n1,2\n3,4").hexdigest()
