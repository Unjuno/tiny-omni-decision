import pytest

from scripts.audit_perception_av_event_task import media_coverage, summarize_split


def sound(sound_id=1):
    return {
        "id": sound_id,
        "label": "Object:sound",
        "parent_objects": [1],
        "is_visible": 1,
        "frame_ids": [0, 10],
    }


def tracks():
    return [
        {"id": 1, "frame_ids": [5], "bounding_boxes": [[0.1, 0.1, 0.3, 0.3]], "is_masked": [0]},
        {"id": 2, "frame_ids": [5], "bounding_boxes": [[0.5, 0.5, 0.8, 0.8]], "is_masked": [0]},
    ]


def test_split_summary_keeps_all_events_and_candidate_source_ids_only_in_audit():
    data = {
        "video_1": {
            "metadata": {"split": "train", "video_id": "video_1"},
            "object_tracking": tracks(),
            "sound_localisation": [sound()],
        }
    }
    result = summarize_split(data, {"video_1": -1}, "train")
    assert result["sound_segments"] == 1
    assert result["candidate_sound_segments"] == 1
    assert result["candidate_videos"] == 1
    assert result["candidate_records"][0]["source_parent_ids"] == [1]
    assert result["candidate_records"][0]["candidate_track_ids"] == [1, 2]


def test_split_summary_fails_if_video_metadata_does_not_match_split_or_identity():
    data = {
        "video_1": {
            "metadata": {"split": "valid", "video_id": "video_1"},
            "object_tracking": tracks(),
            "sound_localisation": [sound()],
        }
    }
    with pytest.raises(ValueError, match="split mismatch"):
        summarize_split(data, {"video_1": -1}, "train")


def test_media_coverage_uses_video_groups_and_counts_only_available_candidates():
    summary = {
        "candidate_video_ids": ["video_1", "video_2"],
        "candidate_records": [
            {"video_id": "video_1", "sound_label": "Object:sound", "candidate_track_ids": [1, 2]},
            {
                "video_id": "video_1",
                "sound_label": "Object:sound",
                "candidate_track_ids": [1, 2, 3],
            },
            {"video_id": "video_2", "sound_label": "Other", "candidate_track_ids": [4, 5]},
        ],
    }
    result = media_coverage(summary, {"video_1"})
    assert result["candidate_videos_in_selected_media"] == 1
    assert result["candidate_events_in_selected_media"] == 2
    assert result["media_video_ids_with_candidate_events"] == ["video_1"]


def test_split_summary_rejects_duplicate_event_ids():
    data = {
        "video_1": {
            "metadata": {"split": "train", "video_id": "video_1"},
            "object_tracking": tracks(),
            "sound_localisation": [sound(), sound()],
        }
    }
    with pytest.raises(ValueError, match="sound segment IDs"):
        summarize_split(data, {"video_1": -1}, "train")
