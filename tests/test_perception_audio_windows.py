import pytest

from tiny_omni_decision.perception_audio_windows import (
    materialize_audio_window,
    split_audio_interval,
)


def test_short_interval_is_center_padded_without_inventing_audio():
    assert split_audio_interval(100_000, 600_000) == [
        {
            "source_start_us": 100_000,
            "source_end_us": 600_000,
            "pad_left_us": 750_000,
            "pad_right_us": 750_000,
        }
    ]


def test_long_interval_is_fully_covered_by_ordered_nonoverlapping_windows():
    windows = split_audio_interval(0, 2_500_000)
    assert windows == [
        {
            "source_start_us": 0,
            "source_end_us": 2_000_000,
            "pad_left_us": 0,
            "pad_right_us": 0,
        },
        {
            "source_start_us": 2_000_000,
            "source_end_us": 2_500_000,
            "pad_left_us": 750_000,
            "pad_right_us": 750_000,
        },
    ]


def test_custom_window_size_preserves_exact_full_window():
    assert split_audio_interval(500, 1_000_500, window_us=1_000_000) == [
        {
            "source_start_us": 500,
            "source_end_us": 1_000_500,
            "pad_left_us": 0,
            "pad_right_us": 0,
        }
    ]


@pytest.mark.parametrize("interval", [(1, 1), (2, 1), (-1, 3)])
def test_invalid_intervals_fail_closed(interval):
    with pytest.raises(ValueError, match="positive half-open"):
        split_audio_interval(*interval)


def test_invalid_window_size_fails_closed():
    with pytest.raises(ValueError, match="positive integer"):
        split_audio_interval(0, 1, window_us=0)


def test_materialized_window_has_fixed_length_and_preserves_audio_samples():
    waveform = [10.0, 20.0]
    window = {
        "source_start_us": 0,
        "source_end_us": 1_000_000,
        "pad_left_us": 500_000,
        "pad_right_us": 500_000,
    }
    assert materialize_audio_window(waveform, window, sample_rate_hz=2) == pytest.approx(
        [0.0, 10.0, 20.0, 0.0]
    )


def test_materialize_rejects_out_of_media_source_interval():
    window = {
        "source_start_us": 1_000_000,
        "source_end_us": 2_000_000,
        "pad_left_us": 500_000,
        "pad_right_us": 500_000,
    }
    with pytest.raises(ValueError, match="outside waveform"):
        materialize_audio_window([1.0, 2.0], window, sample_rate_hz=2)
