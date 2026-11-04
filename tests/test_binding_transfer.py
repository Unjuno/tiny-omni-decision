import numpy as np
import pytest

from scripts.binding_transfer import CONDITIONS, build_conditions, select_states, transfer_gates
from scripts.condition_binding_corpus import all_states, identity


def test_new_states_exclude_every_old_group_deterministically():
    states = all_states()
    excluded = states[:500]
    selected = select_states(excluded, 256)
    assert selected == select_states(list(reversed(excluded)), 256)
    assert len({identity(s) for s in selected}) == 256
    assert not {identity(s) for s in selected} & {identity(s) for s in excluded}
    with pytest.raises(ValueError):
        select_states(states, 1)


def test_all_conditions_keep_same_forty_choices_targets_and_state_groups():
    states = select_states([], 2)
    data = build_conditions(states)
    assert tuple(data) == CONDITIONS
    original = data["known"]
    for name, rows in data.items():
        assert len(rows) == 8 and len({r["id"] for r in rows}) == 8
        for row, base in zip(rows, original, strict=True):
            assert row["target_id"] == base["target_id"]
            assert row["state"] == base["state"] and row["group"] == base["group"]
            assert set(row["option_ids"]) == set(base["option_ids"])
            assert len(row["options"]) == len(set(row["options"])) == 40
            assert row["options"][row["option_ids"].index(row["target_id"])] == row["target"]
            assert row["source"] == "binding" and row["split"] == "development_transfer"
            if name == "no_question":
                assert row["question"] == ""
    for r, normal in zip(data["reverse_joint"], data["joint"], strict=True):
        assert r["question"] == normal["question"]
        assert r["options"] == list(reversed(normal["options"]))
    assert all(
        a["question"] != b["question"] for a, b in zip(data["known"], data["question"], strict=True)
    )


def test_no_question_cannot_answer_two_distinct_fields_and_order_is_id_aligned():
    rows = build_conditions(select_states([], 2))["no_question"]
    from scripts.condition_binding_training import score_summary

    scores = np.tile(np.arange(40), (8, 1))
    result = score_summary(rows, scores)
    assert result["overall"]["accuracy"] <= 0.25
    assert result["pair_both_correct"] == 0


def test_transfer_quality_rule_needs_both_primary_conditions_not_aggregate():
    good = dict(
        overall=dict(accuracy=0.99, nll=0.05, brier=0.02, ece=0.01),
        tasks={f: dict(accuracy=0.99) for f in ("color", "texture", "count", "shape")},
        pair_both_correct=0.98,
    )
    parent = dict(overall=dict(accuracy=0.50, nll=1.0, brier=0.5, ece=0.02))
    summaries = {name: good for name in CONDITIONS}
    references = {name: parent for name in CONDITIONS}
    intervals = {name: dict(lower=[0.40, -1.0]) for name in CONDITIONS}
    assert transfer_gates(summaries, references, intervals) == []
    summaries["joint"] = dict(good, tasks=dict(good["tasks"], color=dict(accuracy=0.94)))
    assert "joint:minimum_task_accuracy" in transfer_gates(summaries, references, intervals)


def test_controls_reject_changed_question_free_distribution_even_same_argmax():
    from scripts.binding_transfer import control_summary

    scores = np.tile(np.arange(40, dtype=float), (8, 1))
    result = control_summary(scores, scores[:, ::-1], scores)
    assert result["same_argmax_identity"]
    assert result["max_question_free_probability_difference"] == 0
    changed = scores.copy()
    changed[1, 39] -= 0.1
    assert np.array_equal(changed.argmax(1), scores.argmax(1))
    with pytest.raises(ValueError, match="question-free"):
        control_summary(scores, scores[:, ::-1], changed)
    with pytest.raises(ValueError, match="order"):
        control_summary(scores, scores, scores)


def test_control_detects_native_tie_break_changing_selected_option_identity():
    from scripts.binding_transfer import control_summary

    scores = np.zeros((4, 40))
    with pytest.raises(ValueError, match="order"):
        control_summary(scores, scores[:, ::-1], scores)
