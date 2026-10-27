import numpy as np
import pytest

from scripts.audit_adl_two_activity_content import content_digest, overlap_summary


def test_hash_normalizes_endianness_and_preserves_content():
    values = np.array([1, -2, 3], dtype="<i2")
    assert content_digest(values)[0] == content_digest(values.astype(">i2"))[0]
    assert content_digest(values)[0] != content_digest(values[:2])[0]


def test_constant_and_empty_content_are_reported():
    assert content_digest(np.array([0, 0], dtype=np.int16))[1:] == (True, False)
    assert content_digest(np.array([], dtype=np.int16))[1:] == (False, True)
    with pytest.raises(ValueError):
        content_digest(np.array([1.0]))


def test_overlap_counts_rows_not_only_unique_hashes():
    result = overlap_summary(["a", "a", "b"], ["a", "c", "a"])
    assert result["common_hashes"] == 1
    assert result["train_rows_with_validation_match"] == 2
    assert result["validation_rows_with_train_match"] == 2
