import pytest

from tiny_omni_decision.habit_decision import (
    habit_transition_points,
    map_habit_choice_target,
)


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


def test_transition_points_cut_inputs_before_target_onset():
    points = habit_transition_points(
        [0.0, 100.0, 200.0, 500.0, 900.0],
        [3, 3, 4, 4, 3],
        pre_transition_guard_ms=100.0,
    )

    assert [(p.target_subtask_id, p.onset_ms, p.context_end_ms) for p in points] == [
        (4, 200.0, 100.0),
        (3, 900.0, 800.0),
    ]
    assert all(point.context_end_ms < point.onset_ms for point in points)


@pytest.mark.parametrize(
    ("times", "labels", "guard"),
    [
        ([], [], 100.0),
        ([0.0, 1.0], [1], 0.1),
        ([0.0, 0.0], [1, 2], 0.1),
        ([0.0, float("nan")], [1, 2], 0.1),
        ([0.0, 1.0], [1, 2], 0.0),
        ([0.0, 1.0], [1, True], 0.1),
    ],
)
def test_invalid_transition_series_fail_closed(times, labels, guard):
    with pytest.raises((TypeError, ValueError)):
        habit_transition_points(times, labels, pre_transition_guard_ms=guard)


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
