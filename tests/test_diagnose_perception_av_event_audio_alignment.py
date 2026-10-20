import importlib
import importlib.util

import pytest


def assign_cross_video_audio_donors(rows):
    name = "scripts.diagnose_perception_av_event_audio_alignment"
    assert importlib.util.find_spec(name) is not None, (
        "audio alignment diagnostic is not implemented"
    )
    audio_alignment = importlib.import_module(name)
    function = getattr(audio_alignment, "assign_cross_video_audio_donors", None)
    assert callable(function), "cross-video audio donor assignment is not implemented"
    return function(rows)


def test_audio_donor_assignment_is_deterministic_and_cross_video():
    rows = [
        {"id": "a-1", "video_id": "video-a"},
        {"id": "a-2", "video_id": "video-a"},
        {"id": "b-1", "video_id": "video-b"},
        {"id": "c-1", "video_id": "video-c"},
    ]

    first = assign_cross_video_audio_donors(rows)
    second = assign_cross_video_audio_donors(list(reversed(rows)))

    assert first == second
    assert set(first) == {row["id"] for row in rows}
    by_id = {row["id"]: row["video_id"] for row in rows}
    assert all(by_id[target_id] != by_id[donor_id] for target_id, donor_id in first.items())


def test_audio_donor_assignment_requires_two_video_groups():
    with pytest.raises(ValueError, match="at least two video groups"):
        assign_cross_video_audio_donors([{"id": "a-1", "video_id": "video-a"}])


def test_audio_donor_assignment_rejects_duplicate_event_ids():
    with pytest.raises(ValueError, match="unique nonempty event IDs"):
        assign_cross_video_audio_donors(
            [
                {"id": "same", "video_id": "video-a"},
                {"id": "same", "video_id": "video-b"},
            ]
        )
