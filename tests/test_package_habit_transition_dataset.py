from __future__ import annotations

import pytest

from scripts.package_habit_transition_dataset import map_transition_text


def test_transition_text_preserves_option_order_and_target() -> None:
    record = {
        "id": "train:1:0",
        "target_subtask_id": 4,
        "target_option_index": 1,
    }
    episode = {
        "high_level_instruction": "Pack the gift set into the box.",
        "low_level_tasks": ["Fold the box.", "Offer the box to the human."],
    }

    mapped = map_transition_text(record, episode, {
        3: "Fold the box.",
        4: "Offer the box to the human.",
    })

    assert mapped["question"] == (
        "Pack the gift set into the box. What should the robot do next?"
    )
    assert mapped["options"] == episode["low_level_tasks"]
    assert mapped["target_index"] == 1


def test_transition_text_fails_if_numeric_target_changed() -> None:
    with pytest.raises(ValueError, match="differs from frozen target"):
        map_transition_text(
            {"id": "train:1:0", "target_subtask_id": 4, "target_option_index": 0},
            {
                "high_level_instruction": "Pack the gift set.",
                "low_level_tasks": ["Fold the box.", "Offer the box."],
            },
            {4: "Offer the box."},
        )


def test_transition_text_collapses_exact_duplicate_actions_stably() -> None:
    mapped = map_transition_text(
        {"id": "train:1:0", "target_subtask_id": 4, "target_option_index": 1},
        {
            "high_level_instruction": "Pack the gift set.",
            "low_level_tasks": ["Fold the box", "Offer it", "offer   it"],
        },
        {4: "Offer it"},
    )

    assert mapped["options"] == ["Fold the box", "Offer it"]
    assert mapped["target_index"] == 1
    assert mapped["collapsed_duplicate_count"] == 1
