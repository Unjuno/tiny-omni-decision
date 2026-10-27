import pytest

pytest.importorskip("torch")
from scripts.train_source_set_nested_threshold import (  # noqa: E402
    choose,
    exact,
    inner_assignment,
    select,
)


def row(ids, targets, probs, none):
    return {
        "candidate_ids": ids,
        "target_candidate_ids": targets,
        "conditional_probabilities": probs,
        "none_probability": none,
    }


def test_group_assignment_is_deterministic_and_disjoint():
    values = ["a", "b", "c", "a", "d"]
    a = inner_assignment(values)
    assert a == inner_assignment(reversed(values))
    assert set(a.values()) == {0, 1}
    with pytest.raises(ValueError):
        inner_assignment(["a", "a"])


def test_decode_none_and_multiple_and_ties():
    r = row(["a", "b"], ["a", "b"], [0.4, 0.4], 0.1)
    assert select(r, 0.5, 0.5) == {"a", "b"}
    assert exact(r, 0.5, 0.5)
    assert select(r, 0.2, 0.05) == {"none_visible"}


def test_selection_uses_all_three_strata_and_is_deterministic():
    rows = [
        row(["a", "b"], [], [0.8, 0.2], 0.8),
        row(["a", "b"], ["a"], [0.8, 0.2], 0.1),
        row(["a", "b"], ["a", "b"], [0.8, 0.4], 0.1),
    ]
    best, grid = choose(rows)
    assert len(grid) == 15 and best["macro_exact"] == 1
    assert choose(rows) == (best, grid)
    with pytest.raises(ValueError):
        choose(rows[:2])
