import copy

import pytest

from scripts.compare_conditional_runs import paired_accuracy


def test_paired_groups_and_integrity():
    rows = [dict(id=str(i), group=str(i // 2), target="a", options=["a", "b"]) for i in range(4)]
    left = [dict(r, probabilities=[0.1, 0.9]) for r in rows]
    right = [dict(r, probabilities=[0.9, 0.1]) for r in rows]
    assert paired_accuracy(rows, left, right) == [1, 1]
    assert paired_accuracy(rows, right, left) == [-1, -1]
    assert paired_accuracy(rows, left, left) == [0, 0]
    with pytest.raises(ValueError):
        paired_accuracy(rows, left[::-1], right)
    for field, value in [
        ("target", "b"),
        ("options", ["b", "a"]),
        ("probabilities", [float("nan"), 0.1]),
        ("probabilities", [0.2, 0.2]),
    ]:
        broken = copy.deepcopy(right)
        broken[0][field] = value
        with pytest.raises(ValueError):
            paired_accuracy(rows, left, broken)
