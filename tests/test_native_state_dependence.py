import copy

import numpy as np
import pytest


def row(i, target, *, task="color", obs=None, options=None):
    options = options or ["red", "blue"]
    return dict(
        id=str(i),
        split="train",
        modality="image",
        source="image",
        task=task,
        observation=obs or str(i),
        group=obs or str(i),
        question="Which color?",
        options=options,
        option_ids=["id-" + x for x in options],
        queries=["query-" + x for x in options],
        target_index=options.index(target),
    )


def test_pairing_matches_full_condition_and_remaps_donor_target():
    from scripts.native_state_dependence import make_plan

    rows = [row(1, "red"), row(2, "blue", options=["blue", "red"])]
    plan = make_plan(rows, {"1", "2"}, limit=8)
    assert len(plan["pairs"]) == 1
    p = plan["pairs"][0]
    assert p["stratum"] == "consumed"
    original = {r["id"]: r for r in rows}
    for ident, target in zip(p["row_ids"], p["targets"], strict=True):
        r = original[ident]
        assert p["options"][target] == r["options"][r["target_index"]]
    assert p["targets"][0] != p["targets"][1]
    assert p["queries"] == ["query-" + x for x in p["options"]]


def test_plan_is_deterministic_order_independent_and_does_not_mutate_rows():
    from scripts.native_state_dependence import make_plan

    rows = [row(i, "red" if i % 2 else "blue") for i in range(30)]
    before = copy.deepcopy(rows)
    used = {str(i) for i in range(10)}
    a = make_plan(rows, used, limit=3)
    assert a == make_plan(list(reversed(rows)), used, limit=3)
    assert rows == before
    assert len(a["pairs"]) == 6
    ids = [i for p in a["pairs"] for i in p["row_ids"]]
    assert len(set(ids)) == len(ids)
    for p in a["pairs"]:
        assert all((i in used) == (p["stratum"] == "consumed") for i in p["row_ids"])


def test_unpairable_tasks_are_retained_and_conditions_never_cross():
    from scripts.native_state_dependence import make_plan

    rows = [
        row(1, "red"),
        dict(row(2, "blue"), question="Different question"),
        row(3, "blue", task="other"),
        row(4, "red", task="other", obs="3"),
    ]
    plan = make_plan(rows, set(), limit=8)
    assert plan["pairs"] == []
    assert len(plan["coverage"]) == 4  # both strata, including zero consumed rows
    assert all(v["selected_pairs"] == 0 for v in plan["coverage"])


@pytest.mark.parametrize("kind", ["validation", "duplicate", "unknown_consumed"])
def test_plan_fails_closed_on_invalid_membership(kind):
    from scripts.native_state_dependence import make_plan

    rows, consumed = [row(1, "red"), row(2, "blue")], set()
    if kind == "validation":
        rows[0]["split"] = "validation"
    elif kind == "duplicate":
        rows.append(rows[0])
    else:
        consumed.add("not-a-training-row")
    with pytest.raises(ValueError):
        make_plan(rows, consumed)


def test_pair_metrics_distinguish_correct_binding_from_merely_changing_predictions():
    from scripts.native_state_dependence import pair_metrics

    p = dict(options=["red", "blue"], targets=[0, 1])
    scores = np.full((1, 2, 40), -1e30)
    scores[0, :, :2] = [[5, 0], [0, 5]]
    good = pair_metrics([p], scores)
    assert good["both_correct"] == good["prediction_changes"] == 1
    assert good["correct_direction_fraction"] == 1
    assert good["symmetric_target_effect"] > 0.98
    wrong = pair_metrics([p], scores[:, ::-1].copy())
    assert wrong["prediction_changes"] == 1 and wrong["both_correct"] == 0
    assert wrong["correct_direction_fraction"] == 0
    assert wrong["symmetric_target_effect"] < -0.98


def test_constant_answer_has_zero_state_effect_even_if_half_answers_correct():
    from scripts.native_state_dependence import pair_metrics

    p = dict(options=["red", "blue"], targets=[0, 1])
    scores = np.full((1, 2, 40), -1e30)
    scores[0, :, :2] = [[5, 0], [5, 0]]
    r = pair_metrics([p], scores)
    assert r["individual"]["accuracy"] == 0.5
    assert r["both_correct"] == r["prediction_changes"] == 0
    assert r["symmetric_target_effect"] == r["probability_total_variation"] == 0


def test_pair_metrics_reject_invalid_targets_padding_and_nonfinite_values():
    from scripts.native_state_dependence import pair_metrics

    p = dict(options=["red", "blue"], targets=[0, 0])
    with pytest.raises(ValueError):
        pair_metrics([p], np.zeros((1, 2, 40)))
    p["targets"] = [0, 1]
    with pytest.raises(ValueError):
        pair_metrics([p], np.zeros((1, 2, 40)))
    scores = np.full((1, 2, 40), -1e30)
    scores[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        pair_metrics([p], scores)


def test_inconsistent_query_features_fail_closed():
    from scripts.native_state_dependence import make_plan

    a, b = row(1, "red"), row(2, "blue")
    b["queries"][0] = "wrong-feature"
    with pytest.raises(ValueError, match="query features"):
        make_plan([a, b], set())


def test_empty_task_stratum_stays_explicit_in_summary():
    from scripts.native_state_dependence import make_plan, summarize

    rows = [row(1, "red"), row(2, "blue")]
    plan = make_plan(rows, {"1", "2"})
    scores = np.full((1, 2, 40), -1e30)
    scores[0, :, :2] = 0
    result = summarize(plan, scores)
    assert result["tasks"]["unconsumed:image:color"] is None
    assert result["strata"]["unconsumed"] is None
