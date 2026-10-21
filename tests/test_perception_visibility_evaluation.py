import numpy as np
import pytest

from tiny_omni_decision.perception_visibility_evaluation import (
    paired_bootstrap_accuracy_delta,
    score_binary_logits,
)


def test_binary_metrics_are_finite_and_match_known_predictions():
    metrics = score_binary_logits(np.array([-20.0, 20.0]), np.array([0, 1]))
    assert metrics["count"] == 2
    assert metrics["accuracy"] == 1.0
    assert metrics["nll"] < 1e-6
    assert metrics["brier"] < 1e-12
    assert metrics["ece_15_equal_width"] < 1e-6


def test_binary_metrics_reject_shape_mismatch_and_nonfinite_logits():
    with pytest.raises(ValueError, match="matching nonempty"):
        score_binary_logits(np.array([1.0]), np.array([0, 1]))
    with pytest.raises(ValueError, match="finite logits"):
        score_binary_logits(np.array([np.nan]), np.array([0]))


def test_paired_bootstrap_is_deterministic_and_reports_b_minus_a():
    result = paired_bootstrap_accuracy_delta(
        np.array([0, 1, 1, 0]), np.array([1, 1, 1, 0]), seed=17, repetitions=200
    )
    assert result["delta_b_minus_a"] == 0.25
    assert result["seed"] == 17
    assert result["bootstrap_repetitions"] == 200
    assert len(result["interval_95"]) == 2
