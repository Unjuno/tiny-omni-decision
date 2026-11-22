from __future__ import annotations

import pytest

from scripts.compare_massive_ja_bert_ruri70m import paired_group_bootstrap


def _prediction(sample_id: str, group: str, target: int, guess: int) -> dict[str, object]:
    probabilities = [0.001] * 60
    probabilities[guess] = 0.941
    probabilities[target] += 0.001
    scale = sum(probabilities)
    probabilities = [value / scale for value in probabilities]
    return {
        "sample_id": sample_id,
        "source_id": sample_id,
        "content_group_id": group,
        "target_index": target,
        "probabilities": probabilities,
    }


def test_group_bootstrap_aligns_examples_and_reports_paired_differences() -> None:
    bert = [_prediction("a", "g1", 0, 0), _prediction("b", "g2", 1, 1)]
    ruri = [_prediction("a", "g1", 0, 1), _prediction("b", "g2", 1, 0)]

    result = paired_group_bootstrap(bert, ruri, iterations=256, seed=17)

    assert result["rows"] == 2
    assert result["normalized_text_groups"] == 2
    assert result["paired_differences"]["accuracy"]["difference_bert_minus_ruri"] == 1.0
    assert result["selection_affected_validation"] is True
    assert result["independent_test"] is False


def test_group_bootstrap_rejects_misaligned_validation_ids() -> None:
    bert = [_prediction("a", "g1", 0, 0)]
    ruri = [_prediction("different", "g1", 0, 0)]

    with pytest.raises(ValueError, match="misaligned at sample_id"):
        paired_group_bootstrap(bert, ruri, iterations=16)


def test_group_bootstrap_rejects_non_normalized_probabilities() -> None:
    bert = [_prediction("a", "g1", 0, 0)]
    ruri = [_prediction("a", "g1", 0, 0)]
    ruri[0]["probabilities"] = [0.5] * 60

    with pytest.raises(ValueError, match="sum to one"):
        paired_group_bootstrap(bert, ruri, iterations=16)
