import pytest

from scripts.build_conditional_curriculum import build
from scripts.build_vocabulary_transfer import render
from scripts.transfer_factorial import PREFIX, make_cells


def test_factorial_changes_question_only_and_preserves_controls():
    for row in build(167, 1, "development"):
        a, b = render(row, "base-0"), render(row, "base-4")
        cells = make_cells(a, b)
        assert len(cells) == 4
        for c in cells:
            for key in ("state", "options", "target", "oracle", "task", "group"):
                assert c[key] == a[key]
        if row["task"] in ("count", "conditional_selection"):
            assert len({c["question"] for c in cells}) == 1
        else:
            q = {c["variant"]: c["question"] for c in cells}
            assert q["base-generic_alias"] == PREFIX + q["base-generic"]
            assert q["base-domain_alias"] == PREFIX + q["base-domain"]


def test_mismatched_pairs_rejected():
    row = build(167, 1, "development")[0]
    a, b = render(row, "base-0"), render(row, "base-4")
    b["state"] = "[]"
    with pytest.raises(ValueError):
        make_cells(a, b)
