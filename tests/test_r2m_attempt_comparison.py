import math

import pytest

from tiny_omni_decision.r2m_attempt_comparison import (
    binary_metric_summary,
    paired_choice_rows,
)


def test_binary_metric_summary_reports_normalized_metrics():
    metrics = binary_metric_summary([[0.1, 0.9], [0.8, 0.2]], [1, 0])

    assert metrics["examples"] == 2
    assert metrics["accuracy"] == 1.0
    assert metrics["nll"] == pytest.approx((-math.log(0.9) - math.log(0.8)) / 2)


def test_paired_choice_uses_same_trial_probability_for_candidate_ranking():
    rows = [
        {"pair_id": "pair", "target": 1, "anchor_sample_id": "anchor",
         "negative_relation": "within_participant", "probabilities": [0.1, 0.9]},
        {"pair_id": "pair", "target": 0, "anchor_sample_id": "anchor",
         "negative_relation": "within_participant", "probabilities": [0.7, 0.3]},
    ]

    paired = paired_choice_rows(rows)

    assert len(paired) == 1
    assert paired[0]["prediction"] == 1
    assert paired[0]["target"] == 1
    assert paired[0]["probabilities"][1] > 0.5


def test_paired_choice_fails_closed_on_incomplete_or_duplicate_pair_rows():
    row = {"pair_id": "pair", "target": 1, "anchor_sample_id": "anchor",
           "negative_relation": "within_participant", "probabilities": [0.1, 0.9]}
    with pytest.raises(ValueError, match="one row for each candidate"):
        paired_choice_rows([row])
    with pytest.raises(ValueError, match="duplicate target"):
        paired_choice_rows([row, dict(row)])
