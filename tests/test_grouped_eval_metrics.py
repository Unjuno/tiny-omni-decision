from tiny_omni_decision.grouped_eval_metrics import (
    classification_metrics,
    grouped_bootstrap,
    percentile_interval,
)


def test_classification_metrics_report_multiclass_scores() -> None:
    result = classification_metrics([[0.8, 0.2], [0.3, 0.7]], [0, 1])
    assert result["accuracy"] == 1.0
    assert result["nll"] > 0
    assert result["brier"] > 0
    assert result["ece15"] > 0


def test_grouped_bootstrap_keeps_clustered_rows_together_and_is_deterministic() -> None:
    probs = [[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.3, 0.7]]
    targets = [0, 1, 0, 1]
    groups = ["speaker-a", "speaker-a", "speaker-b", "speaker-c"]
    first = grouped_bootstrap(probs, targets, groups, replicates=25, seed=17)
    second = grouped_bootstrap(probs, targets, groups, replicates=25, seed=17)
    assert first == second
    assert len(first["metrics"]["accuracy"]) == 25


def test_paired_bootstrap_reports_candidate_minus_baseline() -> None:
    candidate = [[0.9, 0.1], [0.1, 0.9]]
    baseline = [[0.5, 0.5], [0.5, 0.5]]
    result = grouped_bootstrap(
        candidate, [0, 1], ["a", "b"], replicates=10, baseline_probabilities=baseline
    )
    assert all(value < 0 for value in result["candidate_minus_baseline"]["nll"])


def test_percentile_interval_is_two_sided_95_percent() -> None:
    assert percentile_interval(list(range(100))) == [2.475, 96.52499999999999]
