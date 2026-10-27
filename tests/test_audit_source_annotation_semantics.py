from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
from scripts.audit_source_annotation_semantics import audit_record  # noqa: E402


def test_related_parent_can_be_candidate_with_none_visibility():
    builder = SimpleNamespace(_valid_box=lambda box: box, _sample_frames=lambda frames: frames)
    raw = {
        "sound_localisation": [
            {"id": 1, "frame_ids": [0, 0], "is_visible": 0, "parent_objects": [7, -1]}
        ],
        "object_tracking": [{"id": 7, "frame_ids": [0], "bounding_boxes": [[0, 0, 1, 1]]}],
    }
    record = {
        "group_id": "v",
        "split": "train",
        "media_reference": {"video_id": "v"},
        "provenance": {"sound_annotation_id": 1, "upstream_cut_frame": -1},
        "request": {
            "candidates": [
                {
                    "candidate_id": "option_0",
                    "frame_boxes": [{"frame_id": 0, "bbox_xyxy_normalized": [0, 0, 1, 1]}],
                }
            ]
        },
        "supervision": {"none_visible": True, "target_candidate_ids": []},
    }
    result = audit_record(record, raw, builder)
    assert result == {
        "none": True,
        "has_unannotated_parent": True,
        "has_known_parent": True,
        "known_parent_candidates": 1,
    }
    record["request"]["candidates"][0]["candidate_id"] = "option_1"
    with pytest.raises(ValueError, match="reconstruction"):
        audit_record(record, raw, builder)
