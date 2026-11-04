import copy
from collections import Counter

import pytest

from scripts.build_conditional_curriculum import build
from scripts.build_lexical_range_curriculum import build_rows, reword, template


def test_identity_labels_and_balanced_determinism():
    rows = build(17, 16, "train")
    saved = copy.deepcopy(rows)
    actual = build_rows(rows)
    assert actual == build_rows(rows)
    assert Counter(r["wording_variant"] for r in actual) == {i: 32 for i in range(4)}
    for before, after in zip(rows, actual, strict=True):
        for key in before:
            if key != "question":
                assert before[key] == after[key]
        if after["wording_variant"] == 0:
            assert before["question"] == after["question"]
    assert rows == saved
    assert template("At MOST 123?") == "at most <number>?"
    with pytest.raises(ValueError):
        reword(rows[0], 4)
