import pytest

from tiny_omni_decision.intphys_metrics import (
    binary_auc,
    grouped_paired_bootstrap,
    threshold_metrics,
    validation_tail_threshold,
)


def test_binary_auc_handles_perfect_ordering_and_ties():
    assert binary_auc([0, 1], [0.0, 1.0]) == 1.0
    assert binary_auc([0, 1], [0.0, 0.0]) == 0.5


def test_validation_tail_threshold_uses_possible_only_values():
    assert validation_tail_threshold([0.1, 0.2, 0.3, 0.4], tail_probability=0.25) == 0.4


def test_threshold_metrics_report_classwise_sensitivity_specificity():
    result = threshold_metrics([0, 0, 1, 1], [0.1, 0.8, 0.7, 0.9], threshold=0.5)
    assert result["sensitivity_impossible"] == 1.0
    assert result["specificity_possible"] == 0.5
    assert result["balanced_accuracy"] == 0.75
    assert result["roc_auc"] == 0.75


def test_grouped_paired_bootstrap_resamples_scene_groups():
    labels = [0, 1] * 20
    groups = [f"scene-{index}" for index in range(20) for _ in range(2)]
    learned = [0.0, 1.0] * 20
    persistence = [0.0] * 40
    result = grouped_paired_bootstrap(
        labels,
        learned,
        persistence,
        groups,
        first_threshold=0.5,
        second_threshold=0.5,
        replicates=100,
    )
    assert result["scene_groups"] == 20
    assert result["valid_replicates"] == 100
    assert result["first_minus_second_roc_auc"]["ci95"] == [0.5, 0.5]
    assert result["first_minus_second_balanced_accuracy"]["ci95"] == [0.5, 0.5]


@pytest.mark.parametrize(
    "labels,scores",
    [([0, 0], [0.1, 0.2]), ([0, 1], [0.1, float("nan")])],
)
def test_binary_auc_rejects_invalid_inputs(labels, scores):
    with pytest.raises(ValueError):
        binary_auc(labels, scores)
