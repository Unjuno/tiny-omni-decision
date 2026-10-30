import pytest


def test_train_metadata_audit_uses_only_train_rows_and_finds_timestamp_rule():
    from scripts.audit_perception_av_temporal_metadata_shortcut import (
        audit_train_metadata,
    )

    class TrainOnlyManifest(dict):
        def __getitem__(self, key):
            if key == "validation":
                raise AssertionError("metadata audit must not read validation")
            return super().__getitem__(key)

    rows = [
        {
            "video_id": "v1",
            "video_window_s": [0.0, 2.0],
            "target_index": 1,
            "options": [
                {"audio_window_s": [3.0, 5.0]},
                {"audio_window_s": [0.0, 2.0]},
            ],
        },
        {
            "video_id": "v2",
            "video_window_s": [10.0, 12.0],
            "target_index": 0,
            "options": [
                {"audio_window_s": [10.0, 12.0]},
                {"audio_window_s": [13.0, 15.0]},
            ],
        },
    ]
    report = audit_train_metadata(TrainOnlyManifest(train=rows))

    assert report["split"] == "train_only"
    assert report["example_count"] == 2
    assert report["unique_video_groups"] == 2
    assert report["exact_interval_matches"] == 2
    assert report["nearest_center_correct"] == 2
    assert report["maximum_overlap_correct"] == 2
    assert report["validation_rows_used"] is False


def test_train_metadata_audit_rejects_duplicate_video_groups():
    from scripts.audit_perception_av_temporal_metadata_shortcut import (
        audit_train_metadata,
    )

    row = {
        "video_id": "same-video",
        "video_window_s": [0.0, 1.0],
        "target_index": 0,
        "options": [
            {"audio_window_s": [0.0, 1.0]},
            {"audio_window_s": [2.0, 3.0]},
        ],
    }
    with pytest.raises(ValueError, match="one row per video group"):
        audit_train_metadata({"train": [row, row]})
