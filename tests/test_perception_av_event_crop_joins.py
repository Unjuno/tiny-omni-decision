import pytest

from scripts.audit_perception_av_event_crop_joins import (
    group_planned_events,
    validate_event_crop_join,
)
from scripts.perception_av_event_protocol import deterministic_candidate_order


def _valid_event():
    return {
        "video_id": "video_1",
        "sound_id": 4,
        "frame_interval": [3, 5],
        "selected_frame": 4,
        "candidate_track_ids": [7, 9],
        "source_parent_ids": [9],
    }


def _valid_tracks():
    return {
        7: {"frame_ids": [4], "bounding_boxes": [[0.1, 0.2, 0.3, 0.4]], "is_masked": [0]},
        9: {"frame_ids": [4], "bounding_boxes": [[0.6, 0.5, 0.9, 0.8]], "is_masked": [0]},
    }


def test_event_crop_join_preserves_deterministic_candidate_order_and_separate_target():
    event = _valid_event()
    result = validate_event_crop_join(event, _valid_tracks(), [100, 200])
    order = deterministic_candidate_order("video_1", 4, [7, 9])
    assert result["candidate_count"] == 2
    assert result["source_candidate_index_for_separate_label_join"] == order.index(9)
    assert len(result["pixel_boxes_in_model_input_order"]) == len(order)
    assert result["selected_frame"] == 4


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda event: event.update(selected_frame=6), "event interval"),
        (lambda event: event.update(source_parent_ids=[11]), "included source"),
        (lambda event: event.update(candidate_track_ids=[7, 9, 11]), "absent from annotation"),
    ],
)
def test_event_crop_join_fails_closed_for_invalid_join(mutate, match):
    event = _valid_event()
    mutate(event)
    with pytest.raises(ValueError, match=match):
        validate_event_crop_join(event, _valid_tracks(), [100, 200])


def test_event_crop_join_rejects_masked_candidate_crop():
    tracks = _valid_tracks()
    tracks[7]["is_masked"] = [1]
    with pytest.raises(ValueError, match="unmasked"):
        validate_event_crop_join(_valid_event(), tracks, [100, 200])


def test_planned_event_grouping_excludes_unselected_source_records():
    selected = {"video_id": "video_1", "sound_id": 1}
    outside = {"video_id": "video_2", "sound_id": 2}
    assert group_planned_events([selected, outside], ["video_1"]) == {
        "video_1": [selected]
    }
    with pytest.raises(ValueError, match="unique"):
        group_planned_events([selected], ["video_1", "video_1"])
