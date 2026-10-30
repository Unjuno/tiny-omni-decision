import json

import pytest

from scripts.build_conditional_curriculum import build
from scripts.build_numeric_range_curriculum import affine, apply_by_group


def test_numeric_range_transform_preserves_all_answers_and_order():
    rows = build(71, 16, "train")
    for scale, offset in ((1, 0), (2, 7), (4, 31), (8, 0), (16, 23)):
        for row in rows:
            changed = affine(row, scale, offset)
            assert changed["target"] == row["target"]
            assert changed["options"] == row["options"]
            assert changed["id"] == row["id"] and changed["group"] == row["group"]
            original, transformed = json.loads(row["state"]), json.loads(changed["state"])
            for field in ("cost", "risk", "time"):
                assert sorted(range(4), key=lambda i: original[i][field]) == sorted(
                    range(4), key=lambda i: transformed[i][field]
                )
            if "threshold" in row["oracle"]:
                assert [s["id"] for s in original if s["risk"] <= row["oracle"]["threshold"]] == [
                    s["id"] for s in transformed if s["risk"] <= changed["oracle"]["threshold"]
                ]
            else:
                assert changed["question"] == row["question"]
    with pytest.raises(ValueError):
        affine(rows[0], 0, 0)


def test_group_assignment_balanced_deterministic_and_no_replication():
    rows = build(71, 16, "train")
    transforms = [(1, 0), (2, 7), (4, 31), (8, 0)]
    result = apply_by_group(rows, transforms, 71)
    assert result == apply_by_group(rows, transforms, 71)
    assert len(result) == len({r["id"] for r in result}) == len(rows)
    for group in {r["group"] for r in result}:
        assert (
            len({tuple(r["numeric_transform"].values()) for r in result if r["group"] == group})
            == 1
        )
