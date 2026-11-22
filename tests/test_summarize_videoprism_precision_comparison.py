from __future__ import annotations

import pytest

from scripts.pretrained_reuse.summarize_videoprism_precision_comparison import paired_summary


def row(image_id, taxonomy="color", probabilities=(0.8, 0.2)):
    return {
        "image_id": image_id,
        "taxonomy": taxonomy,
        "target": "red",
        "options": ["red", "blue"],
        "probabilities": list(probabilities),
    }


def test_paired_summary_reports_precision_delta_and_taxonomy_metrics():
    baseline = [row("a"), row("b", "shape", (0.2, 0.8))]
    candidate = [row("a", probabilities=(0.7, 0.3)), row("b", "shape", (0.3, 0.7))]

    result = paired_summary(baseline, candidate)

    assert result["baseline"]["count"] == 2
    assert result["argmax_changed_decisions"] == 0
    assert result["correctness_changed_decisions"] == 0
    assert result["max_probability_absolute_delta"] == pytest.approx(0.1)
    assert result["per_taxonomy"]["color"]["baseline"]["accuracy"] == 1.0


def test_paired_summary_rejects_changed_option_order():
    baseline = [row("a")]
    candidate = [
        {**row("a"), "options": ["blue", "red"], "probabilities": [0.2, 0.8]}
    ]

    with pytest.raises(ValueError, match="target/options changed"):
        paired_summary(baseline, candidate)


def test_paired_summary_rejects_non_normalized_probabilities():
    baseline = [row("a")]
    candidate = [row("a", probabilities=(0.5, 0.2))]

    with pytest.raises(ValueError, match="normalized distributions"):
        paired_summary(baseline, candidate)
