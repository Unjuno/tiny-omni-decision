"""Audio-window utilities for source-candidate frozen feature extraction."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from tiny_omni_decision.perception_audio_windows import materialize_audio_window


def materialize_audio_window_at_offset(
    waveform: np.ndarray,
    window: Mapping[str, int],
    *,
    waveform_start_us: int,
    sample_rate_hz: int = 32_000,
) -> np.ndarray:
    """Crop a manifest window from a waveform whose decoded PTS may start >0."""
    if type(waveform_start_us) is not int or waveform_start_us < 0:
        raise ValueError("waveform start PTS must be a nonnegative integer microsecond")
    adjusted = dict(window)
    adjusted["source_start_us"] -= waveform_start_us
    adjusted["source_end_us"] -= waveform_start_us
    if adjusted["source_start_us"] < 0:
        raise ValueError("audio window starts before the decoded waveform PTS")
    return materialize_audio_window(
        waveform, adjusted, sample_rate_hz=sample_rate_hz
    )
