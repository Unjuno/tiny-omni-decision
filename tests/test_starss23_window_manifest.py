import pytest

from scripts.build_starss23_window_manifest import intervals


def test_complete_half_open_intervals_only():
    assert intervals(10, 10) == [(0, 5), (5, 10)]
    assert intervals(10.1, 9.99) == [(0, 5)]
    assert intervals(4.99, 5) == []


@pytest.mark.parametrize("duration", [0, -1, float("nan"), float("inf")])
def test_invalid_duration_rejected(duration):
    with pytest.raises(ValueError):
        intervals(duration, 10)
