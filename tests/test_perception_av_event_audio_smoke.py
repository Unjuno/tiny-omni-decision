import pytest

from scripts.audit_perception_av_event_audio_smoke import (
    event_sample_bounds,
    select_training_event,
)


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


def test_smoke_event_selection_requires_one_exact_train_record():
    rows = [
        {"video_id": "video_1", "sound_id": 3},
        {"video_id": "video_2", "sound_id": 3},
    ]
    assert select_training_event(rows, "video_2", 3) == rows[1]
    with pytest.raises(ValueError, match="exactly one"):
        select_training_event(rows, "video_3", 3)
    with pytest.raises(ValueError, match="exactly one"):
        select_training_event(rows + [rows[1]], "video_2", 3)
