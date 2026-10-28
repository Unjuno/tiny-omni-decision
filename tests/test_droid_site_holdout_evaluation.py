from __future__ import annotations

import pytest

from scripts.evaluate_droid_fresh_next_action import equal_weighted_site_metrics


def _metrics(accuracy: float, ece: float) -> dict[str, float]:
    return {
        "accuracy": accuracy,
        "macro_accuracy": accuracy,
        "nll": 1.0,
        "macro_nll": 1.0,
        "brier": 0.5,
        "ece_10": ece,
    }


def test_equal_weighted_site_metrics_averages_ece_10_and_metrics() -> None:
    per_site = {
        "site_a": {"metrics": {"video": _metrics(0.2, 0.1)}},
        "site_b": {"metrics": {"video": _metrics(0.8, 0.3)}},
    }

    result = equal_weighted_site_metrics(per_site, arms=("video",))

    assert result["video"]["accuracy"] == pytest.approx(0.5)
    assert result["video"]["ece_10"] == pytest.approx(0.2)


def test_equal_weighted_site_metrics_requires_sites() -> None:
    with pytest.raises(ValueError, match="at least one site"):
        equal_weighted_site_metrics({})
