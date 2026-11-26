import json

import numpy as np
import pytest


def test_factorization_preserves_all_options_and_is_normalized():
    from scripts.factorized_condition import combine

    ids = ["x:a", "y:a", "x:b", "y:b"]
    scores = combine([0, 0], [0, 0, 0, np.log(3)], ["x", "y"], ids)
    np.testing.assert_allclose(np.exp(scores), [0.25, 0.125, 0.25, 0.375])
    order = [3, 0, 2, 1]
    permuted = combine(
        [0, 0], np.array([0, 0, 0, np.log(3)])[order], ["x", "y"], [ids[i] for i in order]
    )
    np.testing.assert_array_equal(permuted, scores[order])


def test_factorization_rejects_incomplete_and_nonfinite_scores():
    from scripts.factorized_condition import combine

    for args in [
        ([0, 0], [0], ["x", "y"], ["x:a"]),
        ([0], [np.nan], ["x"], ["x:a"]),
        ([0], [0, 1], ["x"], ["x:a", "x:a"]),
    ]:
        with pytest.raises(ValueError):
            combine(*args)


def test_model_inputs_cannot_depend_on_gold_metadata():
    from scripts.factorized_condition import render

    row = dict(
        state='{"color":"red"}',
        question="Return the color.",
        option_ids=["color:red", "color:blue"],
        options=["The color is red.", "The color is blue."],
    )
    expected = render(row, ["color"])
    poisoned = dict(row, target_id="color:blue", task="unknown", target=1, rule="secret")
    assert render(poisoned, ["color"]) == expected
    assert expected["routing"] == [
        ("Request: Return the color.", "The requested property is color.")
    ]
    assert len(expected["direct"]) == len(expected["values"]) == 2


def test_new_states_exclude_every_previous_identity_deterministically():
    from scripts.condition_binding_corpus import all_states, identity
    from scripts.factorized_condition import select_states

    used = all_states()[:50]
    selected = select_states(used, 32)
    assert len({identity(s) for s in selected}) == 32
    assert not {identity(s) for s in used} & {identity(s) for s in selected}
    assert select_states(list(reversed(used)), 32) == selected
    with pytest.raises(ValueError, match="insufficient"):
        select_states(all_states(), 1)


def test_every_family_has_four_changed_targets_and_forty_options():
    from scripts.condition_binding_corpus import all_states
    from scripts.factorized_condition import make_rows

    rows = make_rows(all_states()[:2])
    assert len(rows) == 64
    assert len({r["id"] for r in rows}) == 64
    for i in range(0, len(rows), 4):
        group = rows[i : i + 4]
        assert len({r["target_id"] for r in group}) == 4
        for row in group:
            assert len(set(row["option_ids"])) == 40
            field, value = row["target_id"].split(":")
            assert str(json.loads(row["state"])[field]) == value


def test_joint_quality_is_not_routing_quality():
    from scripts.factorized_condition import combine
    from scripts.semantic_state_diagnostic import metrics

    score = combine([10.0, -10.0], [0.0, 10.0, 0.0, 0.0], ["x", "y"], ["x:a", "x:b", "y:a", "y:b"])
    assert metrics(score[None], np.array([0]))["accuracy"] == 0


def test_donor_control_changes_facts_and_gold_but_not_request():
    from scripts.condition_binding_corpus import all_states
    from scripts.factorized_condition import control_rows, make_rows

    rows = make_rows([all_states()[0], all_states()[-1]])
    donor = control_rows(rows, "donor")
    assert donor[0]["state"] == rows[4]["state"]
    assert donor[0]["question"] == rows[0]["question"]
    assert donor[0]["target_id"] != rows[0]["target_id"]
    assert control_rows(rows, "reverse")[0]["option_ids"] == rows[0]["option_ids"][::-1]
    assert control_rows(rows, "no_question")[0]["question"] == ""
    assert rows[0]["question"]


def test_research_gate_does_not_accept_high_aggregate_with_weak_family():
    from scripts.factorized_condition import research_failures

    old = dict(
        overall=dict(accuracy=0.8, nll=0.5, brier=0.3, ece=0.02),
        families={
            "positive": {"overall": {"accuracy": 0.8}},
            "negative": {"overall": {"accuracy": 0.8}},
        },
    )
    new = dict(
        overall=dict(accuracy=0.95, nll=0.2, brier=0.1, ece=0.02),
        families={
            "positive": {"overall": {"accuracy": 1}},
            "negative": {"overall": {"accuracy": 0.89}},
        },
    )
    failures = research_failures(old, new, {"lower": [0.02, -0.4]})
    assert "minimum_family_accuracy" in failures
    new["families"]["negative"]["overall"]["accuracy"] = 0.9
    assert research_failures(old, new, {"lower": [0.02, -0.4]}) == []
    assert "paired_accuracy_interval" in research_failures(old, new, {"lower": [0, -0.4]})


def test_order_control_reports_actual_prediction_identity_under_ties():
    from scripts.factorized_condition import order_control

    result = order_control(
        np.array([[0.5, 0.5]]), np.array([[0.5, 0.5]]), [["a", "b"]], [["b", "a"]]
    )
    assert result["max_probability_delta"] == 0
    assert result["identity_agreement"] == 0


def test_replay_accepts_small_float_drift_but_rejects_changed_decisions():
    from scripts.factorized_condition import replay_equal

    baseline = dict(prediction="a", count=4, p=[0.2, 0.8], okay=True)
    assert replay_equal(baseline, dict(baseline, p=[0.2 + 1e-8, 0.8 - 1e-8]))
    assert not replay_equal(baseline, dict(baseline, prediction="b"))
    assert not replay_equal(baseline, dict(baseline, count=5))
    assert not replay_equal(baseline, dict(baseline, p=[0.201, 0.799]))
    assert not replay_equal(baseline, dict(baseline, p=[np.nan, 0.8]))
