from __future__ import annotations

from scripts.compare_jcommonsenseqa_tohoku_bert_readout import paired_group_bootstrap


def _row(sample: str, group: str, target: int, probs: list[float]) -> dict:
    return {
        "sample_id": sample,
        "content_group_id": group,
        "target_index": target,
        "probabilities": probs,
    }


def test_paired_group_bootstrap_preserves_grouping_and_reports_candidate_delta():
    candidate = [
        _row("a", "g1", 0, [0.8, 0.05, 0.05, 0.05, 0.05]),
        _row("b", "g2", 1, [0.05, 0.8, 0.05, 0.05, 0.05]),
    ]
    baseline = [
        _row("a", "g1", 0, [0.2, 0.2, 0.2, 0.2, 0.2]),
        _row("b", "g2", 1, [0.2, 0.2, 0.2, 0.2, 0.2]),
    ]
    result = paired_group_bootstrap(candidate, baseline, iterations=50, seed=17)
    assert result["groups"] == 2
    assert result["candidate"]["accuracy"] == 1.0
    assert result["baseline"]["accuracy"] == 0.5
    assert result["bootstrap"]["candidate_minus_baseline"]["accuracy"]["point_difference"] == 0.5


def test_paired_group_bootstrap_rejects_misaligned_samples():
    candidate = [_row("a", "g1", 0, [1.0, 0, 0, 0, 0])]
    baseline = [_row("other", "g1", 0, [0.2] * 5)]
    try:
        paired_group_bootstrap(candidate, baseline, iterations=2)
    except ValueError as error:
        assert "sample_id" in str(error)
    else:
        raise AssertionError("misaligned sample IDs should fail")
