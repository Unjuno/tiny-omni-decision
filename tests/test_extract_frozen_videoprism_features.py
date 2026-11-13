from __future__ import annotations

import pytest

from scripts.extract_frozen_videoprism_features import (
    _apply_temporal_order,
    validate_split_rows,
)


def test_reverse_temporal_order_preserves_frames_and_reverses_only_order() -> None:
    frames = ["frame-0", "frame-1", "frame-2", "frame-3"]

    reversed_frames = _apply_temporal_order(frames, "reverse")

    assert reversed_frames == ["frame-3", "frame-2", "frame-1", "frame-0"]
    assert sorted(reversed_frames) == sorted(frames)
    assert frames == ["frame-0", "frame-1", "frame-2", "frame-3"]
    assert _apply_temporal_order(reversed_frames, "reverse") == frames


def test_forward_temporal_order_keeps_the_same_sequence() -> None:
    frames = ["frame-0", "frame-1", "frame-2"]

    assert _apply_temporal_order(frames, "forward") is frames


def test_scene_disjoint_templates_can_be_counted_without_weakening_scene_gates() -> None:
    train = [{
        "id": "train-1",
        "split": "train",
        "scene_index": 1,
        "media_sha256": "train-media",
        "question": "Are objects visible?",
        "options": ["yes", "no"],
    }]
    validation = [{
        "id": "validation-10001",
        "split": "validation",
        "scene_index": 10_001,
        "media_sha256": "validation-media",
        "question": "Are objects visible?",
        "options": ["yes", "no"],
    }]

    with pytest.raises(ValueError, match="normalized question content overlaps"):
        validate_split_rows(train, validation)

    assert validate_split_rows(
        train, validation, allow_question_template_overlap=True
    ) == 1


def test_allow_question_template_overlap_does_not_allow_scene_reuse() -> None:
    train = [{
        "id": "train-1",
        "split": "train",
        "scene_index": 1,
        "media_sha256": "same-media",
        "question": "Are objects visible?",
        "options": ["yes", "no"],
    }]
    validation = [{
        "id": "validation-1",
        "split": "validation",
        "scene_index": 1,
        "media_sha256": "same-media",
        "question": "Are objects visible?",
        "options": ["yes", "no"],
    }]

    with pytest.raises(ValueError, match="scene or media identity overlaps"):
        validate_split_rows(train, validation, allow_question_template_overlap=True)
