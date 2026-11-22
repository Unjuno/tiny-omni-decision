from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from analyze_jcola_grouped_predictions import (  # noqa: E402
    compute_metrics,
    paired_source_bootstrap,
)


def _predictions(*, positive_probability: float) -> list[dict[str, object]]:
    rows = []
    for source, targets in (("a", (0, 1)), ("b", (0, 1)), ("c", (0, 1))):
        for index, target in enumerate(targets):
            p1 = positive_probability if target == 1 else 1 - positive_probability
            rows.append(
                {
                    "uid": f"{source}-{index}",
                    "source": source,
                    "target": target,
                    "prediction": int(p1 > 0.5),
                    "p_unacceptable": 1 - p1,
                    "p_acceptable": p1,
                }
            )
    return rows


def test_metrics_detect_majority_collapse_and_calibration() -> None:
    rows = [
        {"uid": "1", "source": "a", "target": 1, "prediction": 1,
         "p_unacceptable": 0.2, "p_acceptable": 0.8},
        {"uid": "2", "source": "a", "target": 0, "prediction": 1,
         "p_unacceptable": 0.2, "p_acceptable": 0.8},
    ]
    result = compute_metrics(rows)
    assert result["accuracy"] == 0.5
    assert result["macro_accuracy"] == 0.5
    assert result["unacceptable_recall"] == 0.0
    assert result["acceptable_recall"] == 1.0
    assert result["mcc"] == 0.0


def test_source_cluster_bootstrap_is_paired_and_deterministic() -> None:
    left = _predictions(positive_probability=0.8)
    right = _predictions(positive_probability=0.9)
    first = paired_source_bootstrap(left, right, repeats=100, seed=17)
    second = paired_source_bootstrap(left, right, repeats=100, seed=17)
    assert first == second
    assert first["source_groups"] == 3
    assert first["metrics"]["accuracy"]["ci95"] == [0.0, 0.0]


def test_paired_bootstrap_rejects_misaligned_ids() -> None:
    with pytest.raises(ValueError, match="different UIDs"):
        paired_source_bootstrap(_predictions(positive_probability=0.8), [], repeats=1)
