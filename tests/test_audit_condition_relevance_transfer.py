import copy
import math

import numpy as np
import pytest

from scripts import audit_condition_relevance_transfer as audit


def test_independent_metric_formulas():
    scores = [[math.log(0.75), math.log(0.25)], [math.log(0.25), math.log(0.75)]]
    metrics, probabilities = audit.independent_metrics(scores, [0, 0])
    assert metrics == pytest.approx(dict(
        count=2, accuracy=0.5, nll=-math.log(0.75 * 0.25) / 2,
        brier=0.625, ece=0.25, mean_confidence=0.75,
    ))
    assert np.allclose(probabilities, [[0.75, 0.25], [0.25, 0.75]], atol=1e-15)


def test_uniform_all40_candidates_and_first_tie():
    metrics, _ = audit.independent_metrics([[0] * 40, [0] * 40], [0, 39])
    assert metrics == pytest.approx(dict(
        count=2, accuracy=0.5, nll=math.log(40), brier=0.975,
        ece=0.475, mean_confidence=0.025,
    ))


@pytest.mark.parametrize("scores,target", [
    ([], []), ([[0, 1]], []), ([[0, math.inf]], [0]),
    ([[0, math.nan]], [0]), ([[0, 1]], [2]), ([[0, 1]], [True]),
])
def test_invalid_metric_inputs_fail(scores, target):
    with pytest.raises(ValueError):
        audit.independent_metrics(scores, target)


def test_extreme_finite_scores_keep_finite_nll():
    metrics, _ = audit.independent_metrics([[0, -10000]], [1])
    assert metrics["nll"] == 10000
    assert metrics["brier"] == 2
    assert metrics["ece"] == 1


def fixture():
    from scripts.condition_relevance_corpus import make_rows
    from scripts.condition_relevance_transfer import transfer_metrics
    from tiny_omni_decision.dataset import CLEVR4_TAXONOMIES

    state = {k: v[0] for k, v in CLEVR4_TAXONOMIES.items()}
    rows = make_rows(state, "transfer", 0)
    scores = np.zeros((len(rows), 40))
    saved = [dict(
        id=r["id"], group=r["group"], condition=r["condition"], field=r["field"],
        target=r["target"], target_id=r["target_id"], option_ids=r["option_ids"],
        prediction=r["options"][0], predicted_id=r["option_ids"][0], probabilities=[0.025] * 40,
    ) for r in rows]
    return rows, scores, saved, transfer_metrics(rows, scores)


def test_saved_predictions_and_all_metric_slices_recompute():
    rows, scores, predictions, summary = fixture()
    result = audit.verify_saved_evidence(rows, scores, predictions, summary)
    assert result["rows"] == 64
    assert result["condition_fields"] == 16
    assert result["maximum_metric_error"] < 1e-12


@pytest.mark.parametrize("change", [
    "id", "group", "condition", "field", "target", "target_id", "option_ids",
    "prediction", "predicted_id", "probabilities", "missing_prediction",
    "metric", "missing_slice", "macro", "minimum", "missing_scores",
])
def test_tampered_saved_outputs_fail(change):
    rows, scores, predictions, summary = fixture()
    predictions = copy.deepcopy(predictions)
    if change in predictions[0]:
        predictions[0][change] = "wrong"
    elif change == "missing_prediction":
        predictions.pop()
    elif change == "metric":
        summary["overall"]["nll"] += 0.1
    elif change == "missing_slice":
        summary["actual_query_slices"].pop(next(iter(summary["actual_query_slices"])))
    elif change == "macro":
        summary["macro_condition_field"]["accuracy"] += 0.1
    elif change == "minimum":
        summary["minimum_condition_field_accuracy"] += 0.1
    elif change == "missing_scores":
        scores = scores[:-1]
    with pytest.raises(ValueError):
        audit.verify_saved_evidence(rows, scores, predictions, summary)


def test_running_result_fails_before_reading_comparison_or_payload(tmp_path, monkeypatch):
    import json

    result = tmp_path / "result.json"
    result.write_text(json.dumps(dict(status="RUNNING")), encoding="utf-8")
    calls = []
    original = audit._json

    def watch(path, expected):
        calls.append(str(path))
        return original(path, expected)

    monkeypatch.setattr(audit, "_json", watch)
    with pytest.raises(ValueError, match="completed fixed transfer"):
        audit.inspect_transfer(tmp_path, audit.sha(result), tmp_path / "absent", "a" * 64)
    assert calls == [str(result)]
