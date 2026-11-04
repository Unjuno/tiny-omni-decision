import pytest

from scripts.plan_text_coverage import coverage_order


def test_new_coverage_is_deterministic_unique_and_balanced():
    original = [f"o{i}" for i in range(20)]
    curriculum = [f"c{i}" for i in range(16)]
    a = coverage_order(original, curriculum, {"o0", "o1"}, steps=3)
    assert a == coverage_order(original, curriculum, {"o0", "o1"}, steps=3)
    assert len(a) == len(set(a)) == 24
    assert not set(a) & {"o0", "o1"}
    for i in range(0, 24, 8):
        assert sum(x.startswith("o") for x in a[i : i + 8]) == 4


def test_insufficient_and_duplicate_rejected():
    with pytest.raises(ValueError):
        coverage_order(["a"], ["b"], set(), steps=1)
    with pytest.raises(ValueError):
        coverage_order(["a"] * 8, ["b"] * 8, set(), steps=1)
