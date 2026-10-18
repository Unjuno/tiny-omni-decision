import pytest

from scripts.audit_perception_av_event_audio_smoke import event_sample_bounds


def test_event_pts_map_to_half_open_sample_interval():
    assert event_sample_bounds(1.25, 1.75, 0.0, 48_000, 96_000) == (60_000, 84_000)


@pytest.mark.parametrize(
    "args",
    [
        (1.0, 1.0, 0.0, 48_000, 96_000),
        (float("nan"), 1.0, 0.0, 48_000, 96_000),
        (-0.1, 0.1, 0.0, 48_000, 96_000),
        (1.0, 2.1, 0.0, 48_000, 96_000),
        (1.0, 2.0, 0.0, 0, 96_000),
    ],
)
def test_invalid_or_out_of_range_event_fails_closed(args):
    with pytest.raises(ValueError):
        event_sample_bounds(*args)

