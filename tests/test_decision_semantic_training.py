"""Contracts for the fixed semantic fit; all fixtures use literal answers."""

import copy
import math

import numpy as np
import pytest

from scripts.decision_semantic_training import (
    accounting,
    bootstrap_minimum,
    learning_rate,
    metric_summary,
    progress_record,
    render_pairs,
    sample_order,
    trainable_names,
)


def row(i, task="binding:4:color", group=None, source="programmatic_decision_v1"):
    return dict(
        id=str(i),
        group=str(group if group is not None else i),
        state='{"color":"red"}',
        question="Choose color.",
        options=["red", "blue"],
        target="red",
        task=task,
        source=source,
    )


def test_rendering_has_no_private_label_or_source_access():
    public = dict(state="plain prose", question="Choose color.", options=["red", "blue"])
    poisoned = dict(
        public,
        target="LEAK",
        rule="LEAK",
        supervision="LEAK",
        source="LEAK",
        task="LEAK",
        option_ids=["LEAK"],
    )
    expected = [
        ("Question: Choose color.\nCandidate answer: red", "State: plain prose"),
        ("Question: Choose color.\nCandidate answer: blue", "State: plain prose"),
    ]
    assert render_pairs(public) == expected == render_pairs(poisoned)


def test_sampler_covers_each_question_once_and_preserves_cohort_counts():
    train = [row(i, "binding:0:color") for i in range(4)]
    train += [row(i + 4, "numeric:count") for i in range(4)]
    replay = [row(8 + i, source="original") for i in range(2)]
    order = sample_order(train, replay, per_cohort=2, replay_per_update=1, seed=17)
    assert order == sample_order(train, replay, per_cohort=2, replay_per_update=1, seed=17)
    assert len(order) == 2 and all(len(b) == 5 for b in order)
    assert sorted(tuple(v) for b in order for v in b) == [(0, i) for i in range(8)] + [
        (1, i) for i in range(2)
    ]
    assert (
        accounting([[train, replay][s][i] for b in order for s, i in b])["repeated_questions"] == 0
    )


def test_sampler_rejects_partial_coverage_and_duplicate_ids():
    rows = [row(0, "binding:0:color"), row(1, "numeric:count")]
    with pytest.raises(ValueError):
        sample_order(rows, [row(3)], per_cohort=2, replay_per_update=1)
    with pytest.raises(ValueError):
        sample_order(rows, [rows[0]], per_cohort=1, replay_per_update=1)


def test_metrics_use_exact_slices_and_macro_not_sample_weighting():
    rows = [row(0), row(1), row(2, "numeric:count")]
    scores = np.array([[4.0, 0.0], [4.0, 0.0], [0.0, 4.0]])
    schema = {"binding:4:color": 2, "numeric:count": 1}
    result = metric_summary(rows, scores, schema)
    assert result["overall"]["accuracy"] == pytest.approx(2 / 3)
    assert result["macro"]["accuracy"] == 0.5
    assert result["minimum_accuracy"] == 0
    assert set(result["metrics"]["numeric:count"]) == {"count", "accuracy", "nll", "brier", "ece"}
    with pytest.raises(ValueError):
        metric_summary(rows, scores, {"binding:4:color": 1, "numeric:count": 2})
    with pytest.raises(ValueError):
        metric_summary(rows, scores[:2], schema)


def test_legacy_metric_key_keeps_numeric_field_and_direction():
    rows = [
        dict(
            row(0, "extremum", source="curriculum"), oracle=dict(field="risk", direction="maximum")
        )
    ]
    result = metric_summary(rows, np.array([[2.0, 0.0]]), {"legacy:curriculum:risk:maximum": 1})
    assert result["minimum_accuracy"] == 1


def test_bootstrap_preserves_whole_states_and_both_cohorts():
    rows = [
        row(f"{cohort}{g}{t}", f"{cohort}:{t}", f"{cohort}{g}")
        for cohort in ("binding", "numeric")
        for g in range(4)
        for t in range(2)
    ]
    wrong = np.tile([0.0, 1.0], (16, 1))
    right = np.tile([1.0, 0.0], (16, 1))
    result = bootstrap_minimum(rows, wrong, right, replicates=100, seed=17)
    assert result["difference"] == 1
    assert result["interval95"] == [1.0, 1.0]
    assert result["states_by_cohort"] == {"binding": 4, "numeric": 4}
    same = bootstrap_minimum(rows, right, right, replicates=100, seed=17)
    assert same["interval95"] == [0.0, 0.0]
    with pytest.raises(ValueError):
        bootstrap_minimum(rows[1:], wrong[1:], right[1:], replicates=10)


def test_accounting_distinguishes_repetition_and_state_count():
    a = row(1)
    b = dict(row(2), state='{"color":"blue"}')
    result = accounting([a, b, copy.deepcopy(a)])
    assert result["consumed"] == 3 and result["unique_questions"] == 2
    assert result["unique_states"] == 2 and result["repeated_questions"] == 1
    assert result["native_assets"] == 0


def test_learning_rate_warmup_and_endpoint_follow_whole_run():
    assert learning_rate(1, 1024, 31, 2e-5) == pytest.approx(2e-5 / 31)
    assert learning_rate(31, 1024, 31, 2e-5) == pytest.approx(2e-5)
    assert 1e-5 < learning_rate(512, 1024, 31, 2e-5) < 2e-5
    assert learning_rate(1024, 1024, 31, 2e-5) == 0
    with pytest.raises(ValueError):
        learning_rate(1025, 1024, 31, 2e-5)


def test_trainable_scope_excludes_embeddings_and_unknown_modules():
    names = [
        "deberta.embeddings.word_embeddings.weight",
        "deberta.encoder.layer.0.weight",
        "deberta.encoder.rel_embeddings.weight",
        "pooler.dense.weight",
        "classifier.bias",
    ]
    assert trainable_names(names) == names[1:]
    with pytest.raises(ValueError):
        trainable_names(names + ["new_head.weight"])


def test_missing_required_metrics_fail_closed():
    scores = np.array([[math.nan, 0.0]])
    with pytest.raises(ValueError):
        metric_summary([row(1)], scores, {"binding:4:color": 1})


def test_padding_cannot_introduce_an_unsupplied_candidate():
    with pytest.raises(ValueError):
        metric_summary([row(1)], np.array([[1.0, 0.0, 2.0]]), {"binding:4:color": 1})


def test_accounting_reports_unique_states_per_source():
    a, b = row(1), dict(row(2, source="original"), state='{"color":"blue"}')
    result = accounting([a, a, b])
    assert result["unique_states_by_source"] == {"programmatic_decision_v1": 1, "original": 1}
    assert result["modalities"] == {"text": 3}


def test_incomplete_progress_never_counts_unexecuted_scheduled_questions():
    rows = [row(i) for i in range(4)]
    order = [[[0, 0], [0, 1]], [[0, 2], [0, 3]]]
    p = progress_record([rows], order, completed_updates=1, forwards=3, backwards=2)
    assert p["optimizer_updates_completed"] == 1
    assert p["completed_question_backwards"] == 2
    assert p["consumed"]["consumed"] == 3
    assert p["partial_update_forwards"] == 1
    assert progress_record([rows], order, 0, 0, 0)["consumed"]["consumed"] == 0
    with pytest.raises(ValueError):
        progress_record([rows], order, 2, 3, 3)
