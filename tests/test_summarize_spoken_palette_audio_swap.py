from __future__ import annotations

import math

import pytest

from scripts.pretrained_reuse.summarize_spoken_palette_audio_swap import (
    decision_metrics,
    summarize,
)


def _records() -> list[dict[str, object]]:
    return [
        {
            "speaker_group": f"speaker-{group:02d}",
            "target_option_id": "a",
            "predicted_option_id": "a",
            "probabilities_by_option_id": {
                "a": 0.7,
                "b": 0.1,
                "c": 0.1,
                "d": 0.1,
            },
        }
        for group in range(24)
        for _ in range(4)
    ]


def test_multiclass_metrics_report_accuracy_loss_brier_and_ece() -> None:
    result = decision_metrics(_records())

    assert result["accuracy"] == 1.0
    assert result["nll"] == pytest.approx(-math.log(0.7))
    assert result["brier_multiclass"] == pytest.approx(0.12)
    assert result["ece_equal_width"] == pytest.approx(0.3)


def test_group_bootstrap_is_deterministic() -> None:
    first = summarize(_records(), replicates=200, seed=17)
    second = summarize(_records(), replicates=200, seed=17)

    assert first == second
    assert first["ci95_percentile"]["accuracy"] == [1.0, 1.0]


def test_metrics_reject_invalid_option_probabilities() -> None:
    rows = _records()
    rows[0]["probabilities_by_option_id"] = {"a": 0.5, "b": 0.5}

    with pytest.raises(ValueError, match="four options"):
        decision_metrics(rows)
