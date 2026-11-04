import numpy as np
import pytest

from tiny_omni_decision.perception_source_candidate_audio import (
    materialize_audio_window_at_offset,
)


def test_audio_window_uses_absolute_pts_and_preserves_center_padding():
    waveform = np.arange(20, dtype=np.float32)
    window = {
        "source_start_us": 1_500_000,
        "source_end_us": 3_000_000,
        "pad_left_us": 250_000,
        "pad_right_us": 250_000,
    }

    result = materialize_audio_window_at_offset(
        waveform, window, waveform_start_us=1_000_000, sample_rate_hz=10
    )

    assert result.shape == (20,)
    assert result[:2].tolist() == [0.0, 0.0]
    assert result[2:17].tolist() == list(map(float, range(5, 20)))
    assert result[17:].tolist() == [0.0, 0.0, 0.0]


def test_audio_window_rejects_uncovered_pre_pts_interval():
    window = {
        "source_start_us": 500_000,
        "source_end_us": 2_000_000,
        "pad_left_us": 250_000,
        "pad_right_us": 250_000,
    }
    with pytest.raises(ValueError, match="before the decoded waveform PTS"):
        materialize_audio_window_at_offset(
            np.zeros(20, dtype=np.float32),
            window,
            waveform_start_us=1_000_000,
            sample_rate_hz=10,
        )
