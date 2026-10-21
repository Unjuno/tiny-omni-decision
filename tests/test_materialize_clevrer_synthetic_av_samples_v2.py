import pytest

from scripts.materialize_clevrer_synthetic_av_samples_v2 import (
    _new_event_rows,
    _validate_pairs,
)


def _annotation():
    return {
        "scene_index": 42,
        "video_filename": "video_00042.mp4",
        "object_property": [
            {"object_id": 0, "color": "blue", "material": "rubber", "shape": "cube"},
            {"object_id": 1, "color": "yellow", "material": "metal", "shape": "sphere"},
        ],
        "collision": [{"object_ids": [0, 1], "frame_id": 50}],
    }


def test_materializes_binary_paired_timing_examples_in_one_scene_split():
    rows = _new_event_rows(
        scene=42,
        split="validation",
        media={"media_path": "media/train/video_00042.mp4", "media_sha256": "f" * 64},
        annotation=_annotation(),
        shift=100,
    )

    assert len(rows) == 2
    assert {row["target_index"] for row in rows} == {0, 1}
    assert {row["split"] for row in rows} == {"validation"}
    assert {row["scene_group_id"] for row in rows} == {"clevrer:train:42"}
    assert {row["sound_onset_frame"] for row in rows} == {134, 166}
    assert {tuple(row["video_frame_range"]) for row in rows} == {(100, 227)}
    _validate_pairs(rows, "validation")


def test_pair_validation_rejects_missing_variant_and_split_mismatch():
    rows = _new_event_rows(
        scene=42,
        split="train",
        media={"media_path": "media/train/video_00042.mp4", "media_sha256": "f" * 64},
        annotation=_annotation(),
        shift=100,
    )
    with pytest.raises(ValueError, match="incomplete paired"):
        _validate_pairs(rows[:1], "train")
    with pytest.raises(ValueError, match="split mismatch"):
        _validate_pairs(rows, "validation")
