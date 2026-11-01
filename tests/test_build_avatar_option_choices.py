from __future__ import annotations

import json

import pytest

from scripts.build_avatar_option_choices import build_choices


def _write_row(root, split, source_id, category, frame=0, fps=30.0):
    clip_id = f"{source_id}_00001"
    folder = root / split / clip_id
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{frame:05d}.json").write_text(
        json.dumps(
            {
                "video_id": f"{source_id}_1",
                "frame_number": frame,
                "fps": fps,
                "duration": 10.0,
                "annotations": [
                    {
                        "task": "Single-Sound",
                        "audio_visual_category": category,
                        "bbox": [1.0, 2.0, 3.0, 4.0],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _fixture(root):
    for split, prefix in (("train", "abcdefghijk"), ("validation", "lmnopqrstuv")):
        for index, category in enumerate(("Guitar", "Piano", "Violin", "Drum")):
            source_id = f"{prefix[:8]}{index:03d}"
            _write_row(root, split, source_id, category)


def test_choices_are_deterministic_group_safe_and_text_free(tmp_path):
    _fixture(tmp_path)

    first = build_choices(tmp_path)
    second = build_choices(tmp_path)

    assert first == second
    train = first["splits"]["train"]
    validation = first["splits"]["validation"]
    assert len(train["examples"]) == 4
    assert len(validation["examples"]) == 4
    all_examples = train["examples"] + validation["examples"]
    assert all(len(row["options"]) == 4 for row in all_examples)
    assert all(row["audio_window_duration_seconds"] == 2.0 for row in all_examples)
    assert all(
        row["audio_window_left_padding_seconds"] == 1.0
        for row in all_examples
        if row["audio_window_center_seconds"] == 0
    )
    assert all(
        option["video_timestamp_seconds"] == 0.0
        for row in all_examples
        for option in row["options"]
    )
    assert all("category" not in json.dumps(row) for row in all_examples)
    uniqueness_keys = (
        "candidate_categories_unique_within_row",
        "candidate_source_groups_unique_within_row",
    )
    assert all(train["report"][key] for key in uniqueness_keys)
    assert sum(train["report"]["target_option_position_counts"].values()) == 4


def test_choices_reject_train_validation_source_group_overlap(tmp_path):
    _write_row(tmp_path, "train", "abcdefghijk", "Guitar")
    _write_row(tmp_path, "validation", "abcdefghijk", "Piano")

    with pytest.raises(ValueError, match="source-video groups overlap"):
        build_choices(tmp_path)


def test_choice_timestamps_use_annotation_fps(tmp_path):
    source = "abcdefghijk"
    _write_row(tmp_path, "train", source, "Guitar", frame=24, fps=29.97002997002997)
    for index, category in enumerate(("Piano", "Violin", "Drum")):
        _write_row(
            tmp_path,
            "train",
            f"bcdefghi{index:03d}",
            category,
            frame=30,
            fps=25.0,
        )
    for index, category in enumerate(("Guitar", "Piano", "Violin", "Drum")):
        _write_row(tmp_path, "validation", f"cdefghij{index:03d}", category)

    result = build_choices(tmp_path)
    train_row = next(
        row for row in result["splits"]["train"]["examples"]
        if row["audio_asset_key"] == f"{source}_00001/00024"
    )

    assert train_row["audio_window_center_seconds"] == pytest.approx(0.8008)


def test_choices_skip_ambiguous_and_mixed_sound_rows(tmp_path):
    categories = ("Guitar", "Piano", "Violin", "Drum", "Guitar")
    for split, prefix in (("train", "abcdefghijk"), ("validation", "lmnopqrstuv")):
        for index, category in enumerate(categories):
            source_id = f"{prefix[:8]}{index:03d}"
            _write_row(tmp_path, split, source_id, category)
    mixed_path = next((tmp_path / "train").rglob("*.json"))
    row = json.loads(mixed_path.read_text(encoding="utf-8"))
    row["annotations"].append(
        {
            "task": "Mixed-Sound",
            "audio_visual_category": "Speech",
            "bbox": [10.0, 10.0, 3.0, 4.0],
        }
    )
    mixed_path.write_text(json.dumps(row), encoding="utf-8")

    result = build_choices(tmp_path)

    assert len(result["splits"]["train"]["examples"]) == 4
