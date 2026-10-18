import numpy as np
import pytest


def test_crop_is_exact_half_second_and_records_sample_rounding():
    from scripts.greatest_hits_inputs import crop_audio

    pcm = np.arange(192000, dtype=np.float32)
    cropped, info = crop_audio(pcm, 96000, 0.0, 1.000003)
    assert cropped.shape == (48000,) and cropped.dtype == np.float32
    np.testing.assert_array_equal(cropped, pcm[72000:120000])
    assert info["start_seconds"] == 0.75 and info["end_seconds"] == 1.25
    assert abs(info["center_error_seconds"]) <= 0.5 / 96000
    with pytest.raises(ValueError):
        crop_audio(pcm, 96000, 0.0, 0.1)
    with pytest.raises(ValueError):
        crop_audio(pcm, 16000, 0.0, 1.0)


def test_video_selection_uses_real_pts_inside_audio_window():
    from scripts.greatest_hits_inputs import select_video

    pts = np.arange(60, dtype=np.float64) * 1001 / 30000
    selected = select_video(pts, 0.75, 1.25)
    assert len(selected) == len(set(selected)) == 8
    assert all(0.75 <= pts[i] < 1.25 for i in selected)
    assert selected == select_video(pts, 0.75, 1.25)
    with pytest.raises(ValueError):
        select_video(pts, 0.75, 0.76)
    with pytest.raises(ValueError):
        select_video(pts[::-1], 0.75, 1.25)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_nonfinite_source_or_times_fail(bad):
    from scripts.greatest_hits_inputs import crop_audio, select_video

    pcm = np.zeros(96000, dtype=np.float32)
    pcm[0] = bad
    with pytest.raises(ValueError):
        crop_audio(pcm, 96000, 0.0, 0.5)
    with pytest.raises(ValueError):
        select_video(np.arange(60, dtype=np.float64) / 30, bad, 1.0)


def test_bad_resampler_input_fails_before_optional_decoder_import():
    from scripts.greatest_hits_inputs import resample_direct

    with pytest.raises(ValueError):
        resample_direct(np.zeros(16000, dtype=np.float32), 32000)
    with pytest.raises(ValueError):
        resample_direct(np.zeros(48000, dtype=np.float32), 32000.0)
