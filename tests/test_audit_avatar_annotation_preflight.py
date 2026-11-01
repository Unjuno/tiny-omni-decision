from __future__ import annotations

import json

import pytest

from scripts.audit_avatar_annotation_preflight import audit_annotations


def _write_frame(root, split, video_id, frame_number, annotations):
    clip_folder = f"{video_id.rsplit('_', 1)[0]}_{int(video_id.rsplit('_', 1)[1]):05d}"
    folder = root / split / clip_folder
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{frame_number:05d}.json").write_text(
        json.dumps(
            {
                "video_id": video_id,
                "frame_number": frame_number,
                "original_width": 640,
                "original_height": 360,
                "annotations": annotations,
            }
        ),
        encoding="utf-8",
    )


def _visible(category="Guitar"):
    return {
        "bbox": [10.0, 10.0, 20.0, 20.0],
        "task": "Single-Sound",
        "audio_visual_category": category,
    }


def test_preflight_reads_train_validation_only_and_reports_temporal_support(tmp_path):
    root = tmp_path / "annotations"
    for frame, category in enumerate(("Guitar", "Piano", "Guitar", "Piano")):
        _write_frame(root, "train", "abcdefghijk_72", frame, [_visible(category)])
    _write_frame(root, "validation", "lmnopqrstuv_3", 0, [_visible()])
    audit = root / "audit" / "deliberately-invalid"
    audit.mkdir(parents=True)
    (audit / "00000.json").write_text("not-json", encoding="utf-8")

    result = audit_annotations(root, check_pins=False)

    assert result["audit_files_opened"] is False
    assert result["splits"]["train"]["potential_temporal_multicategory_clips"] == 1
    assert result["splits"]["train"]["positive_bbox_counts_by_task"] == {"Single-Sound": 4}


def test_preflight_never_opens_audit_annotation_files(tmp_path):
    root = tmp_path / "annotations"
    _write_frame(root, "train", "abcdefghijk_72", 0, [_visible()])
    _write_frame(root, "validation", "lmnopqrstuv_3", 0, [_visible()])
    audit = root / "audit" / "deliberately-invalid"
    audit.mkdir(parents=True)
    (audit / "00000.json").write_text("not-json", encoding="utf-8")

    result = audit_annotations(root, check_pins=False)

    assert result["audit_files_opened"] is False
    assert "audit" not in result["splits"]


def test_preflight_rejects_clip_to_video_id_mismatch(tmp_path):
    root = tmp_path / "annotations"
    _write_frame(root, "train", "abcdefghijk_72", 0, [_visible()])
    path = next((root / "train").rglob("*.json"))
    frame = json.loads(path.read_text(encoding="utf-8"))
    frame["video_id"] = "abcdefghijk_99"
    path.write_text(json.dumps(frame), encoding="utf-8")
    _write_frame(root, "validation", "lmnopqrstuv_3", 0, [_visible()])

    with pytest.raises(ValueError, match="video_id does not match"):
        audit_annotations(root, check_pins=False)
