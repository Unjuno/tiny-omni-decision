import numpy as np
import pytest

from scripts.fit_avatar_native_localization import (
    _paired_group_bootstrap,
    localization_metrics,
)


def test_localization_metrics_handle_multiple_bbox_target_cells():
    logits = np.zeros((1, 10), dtype=np.float64)
    targets = np.zeros((1, 10), dtype=bool)
    targets[0, [0, 1]] = True
    metrics, probabilities = localization_metrics(logits, targets)

    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert metrics["accuracy_hit_at_1"] == 1.0
    assert metrics["hit_at_5"] == 1.0
    assert metrics["nll_target_set"] == np.log(5)
    assert metrics["brier"] == pytest.approx(0.4)
    assert metrics["ece_top1_correctness"] == pytest.approx(0.9)


def test_paired_group_bootstrap_is_deterministic_for_identical_predictions():
    probabilities = np.asarray([[0.8, 0.2], [0.2, 0.8], [0.7, 0.3], [0.1, 0.9]])
    targets = np.asarray([[True, False], [False, True], [True, False], [False, True]])
    groups = ["a", "a", "b", "c"]

    result = _paired_group_bootstrap(probabilities, probabilities, targets, groups, repetitions=100)

    assert result["point_estimate"] == 0.0
    assert result["ci95_percentile"] == [0.0, 0.0]
    assert result["repetitions"] == 100
    assert result["validation_source_group_count"] == 3
