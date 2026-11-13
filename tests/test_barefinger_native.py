import numpy as np
import pytest


def test_full_timestamp_accounting_reports_duplicates_and_reverse_without_retiming():
    from scripts.barefinger_native import summarize_stream

    times = np.array([10, 10.1, 10.1, 10.05, 10.2], dtype=np.float64)[:, None]
    values = np.arange(20, dtype=np.float64).reshape(5, 4)
    before = times.copy(), values.copy()
    report = summarize_stream(times, values)
    assert report["timestamp_duplicates"] == 1
    assert report["timestamp_backwards"] == 1
    assert report["rows"] == 5
    assert report["channels"] == 4
    assert report["value_nonfinite"] == [0, 0, 0, 0]
    np.testing.assert_array_equal(times, before[0])
    np.testing.assert_array_equal(values, before[1])


def test_nonfinite_values_remain_visible_in_source_audit():
    from scripts.barefinger_native import summarize_stream

    values = np.ones((4, 2), dtype=np.float64)
    values[1, 0] = np.nan
    report = summarize_stream(np.arange(4, dtype=np.float64)[:, None], values)
    assert report["value_nonfinite"] == [1, 0]
    assert report["channel_minimum"][0] is None
    assert report["channel_minimum"][1] == 1


@pytest.mark.parametrize(
    "times,values",
    [
        (np.arange(3), np.ones((3, 2))),
        (np.ones((3, 1)), np.ones((4, 2))),
        (np.array([[0.0], [np.nan]]), np.ones((2, 2))),
    ],
)
def test_bad_shapes_or_timestamp_values_rejected(times, values):
    from scripts.barefinger_native import summarize_stream

    with pytest.raises(ValueError):
        summarize_stream(times, values)
