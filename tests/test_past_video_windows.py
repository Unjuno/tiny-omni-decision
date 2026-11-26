import pytest

from tiny_omni_decision.past_video_windows import past_window_indices


def test_no_future_and_endpoint_selection():
    times = list(range(0, 10100, 100))
    chosen = past_window_indices(times, start_ms=1000, end_ms=4000)
    assert len(set(chosen)) == 8
    assert times[chosen[0]] == 1000 and times[chosen[-1]] == 4000
    assert all(1000 <= times[i] <= 4000 for i in chosen)
    assert chosen == past_window_indices(times[:41], start_ms=1000, end_ms=4000)


def test_insufficient_or_invalid_times_fail_closed():
    with pytest.raises(ValueError, match="eight"):
        past_window_indices([0, 1], start_ms=0, end_ms=1)
    for times in ([0, 1, 1], [0, float("nan")], [2, 1]):
        with pytest.raises(ValueError, match="timestamps"):
            past_window_indices(times, start_ms=0, end_ms=4)
    with pytest.raises(ValueError, match="bounds"):
        past_window_indices(range(10), start_ms=4, end_ms=4)
