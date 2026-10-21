import pytest

from scripts.audit_perception_av_error_strata import summarize_source_support, summarize_strata


def test_summarize_strata_reports_support_and_target_heterogeneity():
    rows = [
        {
            "source_label": "Animal:sound",
            "target": "not_visible",
            "audio_prediction": "not_visible",
            "video_prediction": "visible",
            "joint_prediction": "not_visible",
        },
        {
            "source_label": "Animal:sound",
            "target": "not_visible",
            "audio_prediction": "visible",
            "video_prediction": "not_visible",
            "joint_prediction": "visible",
        },
        {
            "source_label": "Human:Speech",
            "target": "visible",
            "audio_prediction": "visible",
            "video_prediction": "visible",
            "joint_prediction": "visible",
        },
    ]
    result = summarize_strata(rows)
    assert result["examples"] == 3
    animal = result["source_label_strata"]["Animal:sound"]
    assert animal["examples"] == 2
    assert animal["visible"] == 0
    assert not animal["contains_both_targets"]
    assert animal["accuracy"] == {"audio": 0.5, "video": 0.5, "joint": 0.5}


def test_summarize_strata_rejects_unknown_target_or_prediction():
    row = {
        "source_label": "x",
        "target": "unknown",
        "audio_prediction": "not_visible",
        "video_prediction": "not_visible",
        "joint_prediction": "not_visible",
    }
    with pytest.raises(ValueError, match="target"):
        summarize_strata([row])


def test_source_support_counts_video_groups_not_just_segments():
    data = {
        "video-a": {
            "sound_localisation": [
                {"label": "speech", "is_visible": 1},
                {"label": "speech", "is_visible": 1},
                {"label": "animal", "is_visible": 0},
                {"label": "animal", "is_visible": 1},
            ]
        },
        "video-b": {
            "sound_localisation": [
                {"label": "speech", "is_visible": 0},
                {"label": "animal", "is_visible": 0},
            ]
        },
    }

    result = summarize_source_support(data)

    assert result["videos"] == 2
    speech = result["source_label_strata"]["speech"]
    assert speech["segments"] == 3
    assert speech["videos"] == 2
    assert speech["visible_videos"] == 1
    assert speech["not_visible_videos"] == 1
    assert speech["videos_with_both_labels"] == 0
    assert result["videos_with_both_labels_any_category"] == 1
