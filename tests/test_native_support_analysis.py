"""CPU-only regression checks for saved positive-support result analysis."""

import numpy as np
import pytest

from scripts.analyze_native_support import (
    event_condition_diagnostic,
    non_speech_balanced,
    recording_bootstrap,
)


def test_recording_bootstrap_paired_identity_is_exact_zero():
    rows = [
        dict(source="starss", observation="starss:record-a:0"),
        dict(source="starss", observation="starss:record-a:1"),
        dict(source="starss", observation="starss:record-b:0"),
        dict(source="lombard", observation="utterance"),
    ]
    vectors = {name: np.array([0.0, 1.0, 2.0, 3.0]) for name in ("accuracy", "nll", "brier")}
    result = recording_bootstrap(rows, vectors, vectors)
    assert result["recordings"] == 2 and result["questions"] == 3
    for name in vectors:
        assert result[name] == dict(candidate_minus_control=0.0, low=0.0, high=0.0)


def test_recording_bootstrap_preserves_recording_clusters_and_question_weighting():
    rows = [
        dict(source="starss", observation="starss:record-a:0"),
        dict(source="starss", observation="starss:record-a:1"),
        dict(source="starss", observation="starss:record-b:0"),
        dict(source="lombard", observation="utterance"),
    ]
    a = {name: np.zeros(4) for name in ("accuracy", "nll", "brier")}
    b = {name: np.array([1.0, 1.0, -1.0, 1000.0]) for name in a}
    result = recording_bootstrap(rows, a, b)
    assert result == recording_bootstrap(rows, a, b)
    # Cluster draws are AA, AB/BA, BB; the two A questions cannot split apart.
    for name in a:
        assert result[name]["candidate_minus_control"] == pytest.approx(1 / 3)
        assert result[name]["low"] == -1
        assert result[name]["high"] == 1


def test_recording_bootstrap_rejects_one_cluster():
    rows = [dict(source="starss", observation="starss:record-a:0")]
    with pytest.raises(ValueError, match="two STARSS"):
        recording_bootstrap(rows, {}, {})


def test_non_speech_balance_keeps_unsupported_bell_and_marks_undefined_water_tap():
    metrics = dict(
        event_classes={
            "bell": dict(balanced_accuracy=0.5),
            "clapping": dict(balanced_accuracy=0.7),
            "speech": dict(balanced_accuracy=0.99),
            "water tap": dict(balanced_accuracy=None),
        }
    )
    result = non_speech_balanced(metrics)
    assert result["mean"] == pytest.approx(0.6)
    assert result["included"] == ["bell", "clapping"]
    assert result["undefined"] == ["water tap"]


def test_event_condition_diagnostic_distinguishes_ranking_and_thresholded_answers():
    rows = [
        dict(
            source="starss",
            observation=f"record:{window}",
            task=str(event),
            options=["no", "yes"],
            target_index=int(event == window),
        )
        for window in range(2)
        for event in range(3)
    ]
    yes = np.array([0.9, 0.1, 0.2, 0.8, 0.9, 0.1])
    scores = np.full((6, 40), -1e30)
    scores[:, :2] = np.log(np.column_stack((1 - yes, yes)))
    r = event_condition_diagnostic(rows, scores)
    assert r["observations"] == 2 and r["classes"] == 3
    assert r["all_event_questions_accuracy"] == 0.5
    assert r["mixed_target_uniform_prediction_fraction"] == 0
    assert r["mixed_target_mean_event_ranking_auc"] == 1
    scores[:, :2] = np.log([0.1, 0.9])
    r = event_condition_diagnostic(rows, scores)
    assert r["mixed_target_uniform_prediction_fraction"] == 1
    assert r["mixed_target_mean_event_ranking_auc"] == 0.5
    assert r["all_event_questions_accuracy"] == 0


def test_event_condition_diagnostic_rejects_missing_event_question():
    rows = [
        dict(
            source="starss",
            observation=f"record:{window}",
            task=str(event),
            options=["no", "yes"],
            target_index=int(event == window),
        )
        for window in range(2)
        for event in range(3)
    ]
    with pytest.raises(ValueError, match="all event questions"):
        event_condition_diagnostic(rows[:-1], np.zeros((5, 40)))
