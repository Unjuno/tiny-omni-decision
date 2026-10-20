from __future__ import annotations

import numpy as np
import pytest

from scripts.build_clevrer_synthetic_av_timing_features import audio_energy_features


def test_audio_energy_features_preserve_relative_tone_timing() -> None:
    before = audio_energy_features(sound_onset_sample=48 * 625, video_start_frame=0)
    after = audio_energy_features(sound_onset_sample=80 * 625, video_start_frame=0)

    assert before.shape == after.shape == (16, 1)
    assert np.isfinite(before).all() and np.isfinite(after).all()
    assert before.sum() == pytest.approx(1.0)
    assert after.sum() == pytest.approx(1.0)
    assert int(before.argmax()) == 6
    assert int(after.argmax()) == 10
    assert not np.array_equal(before, after)


def test_audio_energy_features_align_sound_to_video_clip_start() -> None:
    shifted = audio_energy_features(sound_onset_sample=(200 + 48) * 625, video_start_frame=200)
    unshifted = audio_energy_features(sound_onset_sample=48 * 625, video_start_frame=0)

    assert np.array_equal(shifted, unshifted)


@pytest.mark.parametrize(
    ("onset", "start"),
    [(-1, 0), (128 * 625, 0), (129 * 625, 1), (True, 0)],
)
def test_audio_energy_features_reject_invalid_onsets(onset: int, start: int) -> None:
    with pytest.raises(ValueError):
        audio_energy_features(sound_onset_sample=onset, video_start_frame=start)
