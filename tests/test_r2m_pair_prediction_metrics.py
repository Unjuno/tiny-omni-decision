import pytest


def paired_choice_summary(*args):
    pytest.importorskip("torch")
    from scripts.train_r2m_gated_decision_pilot import paired_choice_summary as summarize

    return summarize(*args)


def test_paired_choice_metrics_score_positive_candidate_and_keep_relation_strata():
    rows = [
        {"pair_id": "a", "target": 0, "negative_relation": "cross_participant",
         "probabilities": [0.9, 0.1]},
        {"pair_id": "a", "target": 1, "negative_relation": "cross_participant",
         "probabilities": [0.1, 0.9]},
        {"pair_id": "b", "target": 0, "negative_relation": "within_participant",
         "probabilities": [0.8, 0.2]},
        {"pair_id": "b", "target": 1, "negative_relation": "within_participant",
         "probabilities": [0.2, 0.8]},
    ]
    metrics, predictions = paired_choice_summary(rows)
    assert metrics["accuracy"] == 1.0
    assert metrics["examples"] == 2
    assert metrics["by_negative_relation"]["cross_participant"]["examples"] == 1
    assert metrics["by_negative_relation"]["within_participant"]["examples"] == 1
    assert all(row["prediction"] == 1 for row in predictions)


def test_paired_choice_metrics_ignore_incomplete_questions():
    rows = [
        {"pair_id": "only-negative", "target": 0,
         "probabilities": [0.7, 0.3]},
    ]
    metrics, predictions = paired_choice_summary(rows)
    assert metrics["examples"] == 0
    assert metrics["accuracy"] is None
    assert predictions == []
