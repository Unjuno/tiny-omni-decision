import pytest

from tiny_omni_decision.habit_transition import (
    resolve_episode_options,
    transition_window_for_boundary,
)


def test_episode_options_resolve_to_pinned_global_subtask_ids():
    global_subtasks = {
        15: "Hold the roller.",
        16: "Follow the pointing gesture.",
        17: "Clean the other side.",
    }

    options = resolve_episode_options(
        ["Follow the pointing gesture.", "Hold the roller."], global_subtasks
    )

    assert options == ((15, "Hold the roller."), (16, "Follow the pointing gesture."))


def test_episode_options_fail_closed_on_missing_or_ambiguous_text():
    with pytest.raises(ValueError, match="not present"):
        resolve_episode_options(["Unknown action"], {15: "Hold the roller."})
    assert resolve_episode_options(
        ["Same", "Same", "Other"], {15: "Same", 16: "Other"}
    ) == ((15, "Same"), (16, "Other"))
    with pytest.raises(ValueError, match="duplicate global subtask text"):
        resolve_episode_options(
            ["Same", "Other"], {15: "Same", 16: "Same", 17: "Other"}
        )
    assert resolve_episode_options(["Only"], {15: "Only"}) == ((15, "Only"),)


def test_transition_window_uses_only_pretransition_inputs_and_future_target():
    labels = [15] * 20 + [16] * 20

    window = transition_window_for_boundary(
        labels, boundary=20, option_ids=(15, 16), frame_count=8
    )

    assert window is not None
    assert window.history_indices == tuple(range(8, 16))
    assert max(window.history_indices) < window.boundary
    assert window.target_index == 25
    assert window.target_index > window.boundary
    assert window.target_option_id == 16


def test_transition_window_skips_boundary_without_full_context():
    labels = [15] * 7 + [16] * 33
    assert transition_window_for_boundary(
        labels, boundary=7, option_ids=(15, 16), frame_count=8
    ) is None


def test_transition_window_skips_boundary_without_full_future_horizon():
    labels = [15] * 20 + [16] * 2
    assert transition_window_for_boundary(
        labels, boundary=20, option_ids=(15, 16), frame_count=8
    ) is None


def test_transition_window_rejects_nontransition_boundary():
    labels = [15] * 20 + [16] * 20
    with pytest.raises(ValueError, match="does not identify"):
        transition_window_for_boundary(
            labels, boundary=25, option_ids=(15, 16), frame_count=8
        )


def test_transition_window_rejects_target_outside_supplied_options():
    labels = [15] * 20 + [16] * 20
    with pytest.raises(ValueError, match="candidate option"):
        transition_window_for_boundary(
            labels, boundary=20, option_ids=(15,), frame_count=8
        )
