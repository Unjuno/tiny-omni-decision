import pytest

from tiny_omni_decision.temporal_sampling_plan import plan_windows


def test_dense_plan_bounds_and_tail():
    stamps = [i / 30 for i in range(405)]
    result = plan_windows(stamps, start=0, end=404 / 30)
    assert all(0 <= stamps[i] < 404 / 30 for i in result["indices"])
    assert set(result["indices"]) == {i for w in result["windows"] for i in w}
    assert all(len(w) == len(set(w)) == 8 for w in result["windows"])
    assert any(0.4 <= stamps[i] <= 0.8 for i in result["indices"])
    assert result["max_gap_seconds"] <= 0.284
    assert result == plan_windows(stamps, start=0, end=404 / 30)


def test_dense_plan_rejects_bad_clock_and_short_input():
    with pytest.raises(ValueError, match="timestamps"):
        plan_windows([0, 0, 1], start=0, end=1)
    with pytest.raises(ValueError, match="insufficient"):
        plan_windows([0, 0.1], start=0, end=0.2)
