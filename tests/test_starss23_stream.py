import pytest

from scripts.starss23_stream import validate_intervals


@pytest.mark.parametrize(
    "windows", [[], [(1, 5)], [(0, 5), (4, 9)], [(5, 10), (0, 5)], [(float("nan"), 5)]]
)
def test_invalid_window_sequence_rejected(windows):
    with pytest.raises(ValueError):
        validate_intervals(windows)


def test_ordered_contiguous_or_gapped_windows():
    validate_intervals([(0, 5), (5, 10)])
    validate_intervals([(0, 5), (20, 25)])
