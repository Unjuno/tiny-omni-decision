import pytest

from scripts.audit_perception_av_event_audio_coverage import estimate_audio_coverage


def test_audio_coverage_maps_inclusive_frame_interval():
    result = estimate_audio_coverage([10, 19], 10.0, 96_000, 48_000)
    assert result["requested_seconds"] == pytest.approx(1.0)
    assert result["available_seconds"] == pytest.approx(1.0)
    assert result["missing_tail_seconds"] == 0
    assert result["coverage_fraction"] == 1.0


def test_partial_tail_is_measured_without_dropping_window():
    result = estimate_audio_coverage([10, 19], 10.0, 93_600, 48_000)
    assert result["requested_seconds"] == pytest.approx(1.0)
    assert result["available_seconds"] == pytest.approx(0.95)
    assert result["missing_tail_seconds"] == pytest.approx(0.05)
    assert result["coverage_fraction"] == pytest.approx(0.95)


def test_zero_overlap_is_explicit():
    result = estimate_audio_coverage([10, 19], 10.0, 24_000, 48_000)
    assert result["available_seconds"] == 0
    assert result["coverage_fraction"] == 0


@pytest.mark.parametrize(
    "args",
    [
        ([10], 10.0, 48_000, 48_000),
        ([20, 19], 10.0, 48_000, 48_000),
        ([-1, 2], 10.0, 48_000, 48_000),
        ([0, 1], 0.0, 48_000, 48_000),
    ],
)
def test_invalid_interval_or_media_metadata_fails_closed(args):
    with pytest.raises(ValueError):
        estimate_audio_coverage(*args)
