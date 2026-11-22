import pytest

from tiny_omni_decision.haptile_action import (
    ACTION_LABELS,
    ACTION_OPTION_TEXTS,
    directional_action_label,
)


@pytest.mark.parametrize(
    ("future", "expected"),
    [
        ((-0.02, 0.0, 0.0), "-x"),
        ((0.02, 0.0, 0.0), "+x"),
        ((0.0, -0.02, 0.0), "-y"),
        ((0.0, 0.02, 0.0), "+y"),
        ((0.0, 0.0, -0.02), "-z"),
        ((0.0, 0.0, 0.02), "+z"),
        ((0.002, 0.002, 0.002), "hold"),
    ],
)
def test_directional_action_label_covers_six_directions_and_hold(future, expected):
    assert directional_action_label((0.0, 0.0, 0.0), future) == expected


def test_directional_action_label_uses_stable_axis_tie_break():
    assert directional_action_label((0.0, 0.0, 0.0), (0.02, -0.02, 0.0)) == "+x"


def test_action_option_texts_are_unique_and_match_fixed_json_labels():
    assert len(ACTION_OPTION_TEXTS) == len(ACTION_LABELS) == 7
    assert len(set(ACTION_OPTION_TEXTS)) == len(ACTION_OPTION_TEXTS)


@pytest.mark.parametrize(
    ("current", "future", "deadband"),
    [
        ((0.0, 0.0), (1.0, 0.0), 0.005),
        ((0.0, 0.0, 0.0), (float("nan"), 0.0, 0.0), 0.005),
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 0.0),
    ],
)
def test_directional_action_label_rejects_invalid_inputs(current, future, deadband):
    with pytest.raises(ValueError):
        directional_action_label(current, future, deadband_m=deadband)
