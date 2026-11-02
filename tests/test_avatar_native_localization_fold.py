import json

import pytest

from scripts.build_avatar_native_localization_fold import (
    _annotation_record,
    partition_source_groups,
    source_group_for_clip,
)
from scripts.extract_avatar_dinov2_spatial_train import bbox_center_cell, letterbox_frame


def test_source_group_removes_only_final_five_digit_segment_suffix():
    assert source_group_for_clip("-00nar1nEPc_00072") == "-00nar1nEPc"
    assert source_group_for_clip("__ZkQzOL9Ac_00063") == "__ZkQzOL9Ac"
    with pytest.raises(ValueError, match="five-digit"):
        source_group_for_clip("video_without_segment")


def test_partition_is_deterministic_and_disjoint():
    groups = {f"source-{index:04d}" for index in range(40)}
    train_a, validation_a = partition_source_groups(groups)
    train_b, validation_b = partition_source_groups(set(reversed(sorted(groups))))
    assert train_a == train_b
    assert validation_a == validation_b
    assert not train_a & validation_a
    assert train_a | validation_a == groups


def test_annotation_record_keeps_bbox_and_exact_frame_timestamp(tmp_path):
    clip_dir = tmp_path / "source123456_00001"
    clip_dir.mkdir()
    path = clip_dir / "00018.json"
    path.write_text("{}", encoding="utf-8")
    record = _annotation_record(
        path,
        {
            "frame_number": 18,
            "fps": 27.0,
            "original_width": 640,
            "original_height": 360,
            "annotations": [
                {
                    "task": "Single-Sound",
                    "bbox": [100, 50, 120, 80],
                    "audio_visual_category": "Piano",
                }
            ],
        },
    )
    assert record is not None
    assert record["asset_key"] == "source123456_00001/00018"
    assert record["source_group"] == "source123456"
    assert record["timestamp_seconds"] == pytest.approx(18 / 27)
    assert record["bbox_xywhs"] == [[100.0, 50.0, 120.0, 80.0]]


def test_non_single_sound_frame_is_excluded(tmp_path):
    clip_dir = tmp_path / "source123456_00001"
    clip_dir.mkdir()
    path = clip_dir / "00018.json"
    path.write_text(json.dumps({"fixture": True}), encoding="utf-8")
    assert (
        _annotation_record(
            path,
            {"annotations": [{"task": "Mixed-Sound"}]},
        )
        is None
    )


def test_spatial_letterbox_preserves_aspect_and_maps_bbox_center():
    import numpy as np

    frame = np.full((360, 640, 3), 127, dtype=np.uint8)
    canvas, transform = letterbox_frame(frame)
    assert canvas.shape == (140, 224, 3)
    assert transform["scale_x"] == pytest.approx(0.35)
    assert transform["scale_y"] == pytest.approx(0.35)
    assert transform["pad_y"] == 7
    assert bbox_center_cell([[280, 140, 80, 80]], transform) == [20]
