from __future__ import annotations

import numpy as np
import pytest

from scripts.extract_bsd10k_frozen_audio_features import (
    HOP_SAMPLES,
    WINDOW_SAMPLES,
    fixed_audio_windows,
)


def test_short_clip_is_right_padded_to_one_window():
    waveform = np.ones(100, dtype=np.float32)
    windows = fixed_audio_windows(waveform)
    assert windows.shape == (1, WINDOW_SAMPLES)
    assert np.array_equal(windows[0, :100], waveform)
    assert np.count_nonzero(windows[0, 100:]) == 0


def test_long_clip_uses_one_second_hops_and_covers_tail():
    waveform = np.ones(WINDOW_SAMPLES + HOP_SAMPLES // 2, dtype=np.float32)
    windows = fixed_audio_windows(waveform)
    assert windows.shape == (2, WINDOW_SAMPLES)
    valid_tail = waveform[HOP_SAMPLES:]
    assert np.array_equal(windows[-1, : len(valid_tail)], valid_tail)
    assert np.count_nonzero(windows[-1, len(valid_tail) :]) == 0


@pytest.mark.parametrize(
    "waveform",
    [np.array([], dtype=np.float32), np.array([[1.0]], dtype=np.float32), np.array([np.nan])],
)
def test_window_builder_rejects_empty_nonmono_or_nonfinite(waveform):
    with pytest.raises(ValueError, match="waveform"):
        fixed_audio_windows(waveform)
