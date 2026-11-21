import pytest

from tiny_omni_decision.habit_decision import map_habit_choice_target


def test_repeated_action_text_collapses_and_maps_global_target_id():
    choices = [
        "Pick up left box",
        "Offer the box",
        "Release the box",
        "Pick up right box",
        "Offer the box",
        "Release the box",
    ]
    result = map_habit_choice_target(
        choices,
        target_subtask_id=4,
        global_subtasks={
            0: "unused",
            1: "Offer the box",
            2: "Release the box",
            3: "unused",
            4: "Offer the box",
            5: "Release the box",
        },
    )

    assert result.options == (
        "Pick up left box", "Offer the box", "Release the box", "Pick up right box"
    )
    assert result.target_index == 1
    assert result.collapsed_duplicate_count == 2


def test_text_normalization_collapses_spacing_and_unicode_variants():
    result = map_habit_choice_target(
        ["  Café   cup ", "Close box", "Cafe\u0301 cup"],
        target_subtask_id=2,
        global_subtasks={0: "unused", 1: "Close box", 2: "Café cup"},
    )

    assert result.options == ("Café   cup", "Close box")
    assert result.target_index == 0
    assert result.collapsed_duplicate_count == 1


@pytest.mark.parametrize(
    ("options", "target_id", "catalog", "message"),
    [
        (["Only action"], 0, {0: "Only action"}, "two unique"),
        (["First", "Second"], 0, {0: "Missing"}, "absent"),
        (["First", "Second"], 3, {0: "First"}, "outside"),
        (["First", "Second"], True, {0: "First"}, "integer"),
    ],
)
def test_invalid_candidate_target_mappings_fail_closed(options, target_id, catalog, message):
    with pytest.raises((TypeError, ValueError), match=message):
        map_habit_choice_target(options, target_id, catalog)
