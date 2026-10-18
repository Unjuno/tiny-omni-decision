import pytest

from scripts.audit_perception_av_correspondence import event_candidate


def sound(*, parents=(1,), visible=1, frames=(0, 5)):
    return {
        "id": 7,
        "label": "Object:sound",
        "parent_objects": list(parents),
        "is_visible": visible,
        "frame_ids": list(frames),
    }


def objects(*, source_visible=True, competitor_visible=True):
    return [
        {
            "id": 1,
            "frame_ids": [0, 5],
            "bounding_boxes": [[0.1, 0.1, 0.3, 0.3], [0.1, 0.1, 0.3, 0.3]],
            "is_masked": [0 if source_visible else 1, 0 if source_visible else 1],
        },
        {
            "id": 2,
            "frame_ids": [0, 5],
            "bounding_boxes": [[0.5, 0.5, 0.8, 0.8], [0.5, 0.5, 0.8, 0.8]],
            "is_masked": [0 if competitor_visible else 1, 0 if competitor_visible else 1],
        },
    ]


def test_single_visible_parent_with_competitor_is_candidate():
    result = event_candidate(sound(), objects(), cut_frame=10)
    assert result["status"] == "candidate"
    assert result["candidate_frames"] == [
        {"frame_id": 0, "candidate_count": 2},
        {"frame_id": 5, "candidate_count": 2},
    ]


@pytest.mark.parametrize(
    ("event", "tracks", "cut", "reason"),
    [
        (sound(parents=(-1,)), objects(), 10, "unknown_or_unresolved_parent"),
        (
            {**sound(), "label": "Muted"},
            objects(),
            10,
            "muted_audio_label",
        ),
        (sound(parents=(1, 2)), objects(), 10, "multiple_parent_sources"),
        (sound(visible=0), objects(), 10, "source_not_visible"),
        (sound(), objects(source_visible=False), 10, "no_shared_visible_competitor_frame"),
        (sound(frames=(5, 10)), objects(), 10, "crosses_or_reaches_cut"),
    ],
)
def test_ambiguous_or_ineligible_rows_are_held(event, tracks, cut, reason):
    result = event_candidate(event, tracks, cut)
    assert result == {"status": "held", "reason": reason, "candidate_frames": []}


def test_misaligned_track_arrays_fail_closed():
    tracks = objects()
    tracks[0]["is_masked"].pop()
    with pytest.raises(ValueError, match="track annotation lengths differ"):
        event_candidate(sound(), tracks, cut_frame=10)


def test_competitors_at_different_times_do_not_form_a_joint_option_set():
    tracks = objects()
    tracks[0]["frame_ids"] = [0]
    tracks[0]["bounding_boxes"] = [[0.1, 0.1, 0.3, 0.3]]
    tracks[0]["is_masked"] = [0]
    tracks[1]["frame_ids"] = [5]
    tracks[1]["bounding_boxes"] = [[0.5, 0.5, 0.8, 0.8]]
    tracks[1]["is_masked"] = [0]
    assert event_candidate(sound(), tracks, cut_frame=10) == {
        "status": "held",
        "reason": "no_shared_visible_competitor_frame",
        "candidate_frames": [],
    }
