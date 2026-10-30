"""Unit checks for metric behavior used by the fixed-budget native readout diagnostic."""

import pytest

torch = pytest.importorskip("torch")

from scripts.fit_native_feature_inat_diagnostic import metric_summary  # noqa: E402


def test_metric_summary_reports_perfect_multiclass_decisions() -> None:
    logits = torch.tensor([[8.0, 0.0], [0.0, 8.0], [8.0, 0.0], [0.0, 8.0]])
    targets = torch.tensor([0, 1, 0, 1])

    result = metric_summary(logits, targets, ["a", "b"])

    assert result["count"] == 4
    assert result["accuracy"] == 1.0
    assert result["macro_accuracy"] == 1.0
    assert result["brier"] < 1e-6
    assert result["nll"] < 1e-3
    assert result["ece_15_bins"] < 1e-3
    assert result["per_class"]["a"]["count"] == 2
    assert result["per_class"]["b"]["count"] == 2
