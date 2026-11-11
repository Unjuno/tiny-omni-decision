from __future__ import annotations

import pytest

from scripts.verify_path_a_perception_train_runtime_parity import (
    deterministic_group_partition,
    project_runtime_fields,
)


def test_group_partition_is_deterministic_and_keeps_each_video_in_one_side() -> None:
    groups = [f"video-{index // 3:03d}" for index in range(90)]
    fit_a, selection_a = deterministic_group_partition(groups)
    fit_b, selection_b = deterministic_group_partition(groups)

    assert fit_a == fit_b
    assert selection_a == selection_b
    assert not fit_a & selection_a
    assert fit_a | selection_a == set(groups)
    assert len(selection_a) == 5


def test_group_partition_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="source video identity"):
        deterministic_group_partition(["", "video-1"])
    with pytest.raises(ValueError, match="selection_fraction"):
        deterministic_group_partition(["video-1", "video-2"], selection_fraction=1.0)


def test_runtime_projection_excludes_labels_and_targets() -> None:
    projected = project_runtime_fields(
        {
            "id": "sample-1",
            "source_video_id": "video-1",
            "question": "Which event occurs?",
            "options": ["A: one", "B: two", "C: three"],
            "reasoning_type": "descriptive",
            "target": "A: one",
            "target_option_index": 0,
            "label": "leak-check",
        }
    )

    assert set(projected) == {
        "id",
        "source_video_id",
        "question",
        "options",
        "reasoning_type",
    }
    assert "target" not in projected
    assert "target_option_index" not in projected
    assert "label" not in projected
