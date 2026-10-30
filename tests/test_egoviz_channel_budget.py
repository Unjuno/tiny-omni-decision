import numpy as np
import pytest

from scripts.egoviz_channel_budget_policy import evaluation_epochs, improves
from scripts.verify_egoviz_channel_budget import (
    assert_logits_match,
    check_predictions,
    compare_predictions,
)


def test_budget_and_checkpoint_selection_are_distinct():
    epochs = evaluation_epochs(512)
    assert epochs[0] == 1
    assert {32, 64, 128, 256, 384, 512}.issubset(epochs)
    assert epochs[-1] == 512
    assert improves(1.0, 1.1)
    assert not improves(1.0, 1.0000005)
    assert not improves(1.1, 1.0)
    with pytest.raises(ValueError, match="nonfinite"):
        improves(np.nan, 1.0)


def test_paired_comparison_rejects_mismatched_sample_identity():
    baseline = [{"sample_id": "a", "group_id": "g", "target": 0, "logits": [1, 0, 0, 0]}]
    candidate = [dict(baseline[0], sample_id="other")]
    with pytest.raises(ValueError, match="sample_id"):
        compare_predictions(baseline, candidate)


def test_verification_rejects_wrong_group_and_nonfinite_reload():
    saved = [{"sample_id": "a", "group_id": "wrong", "target": 0, "logits": [1, 0, 0, 0]}]
    with pytest.raises(ValueError, match="group_id"):
        check_predictions(saved, ["a"], ["expected"], [0])
    with pytest.raises(ValueError, match="nonfinite"):
        assert_logits_match(np.array([[np.nan, 0, 0, 0]]), np.array([[1, 0, 0, 0]]))


def test_paired_accuracy_difference_has_the_correct_direction():
    pytest.importorskip("torch")
    baseline = [
        {"sample_id": str(i), "group_id": f"g{i}", "target": 0, "logits": [2, 0, 0, 0]}
        for i in range(20)
    ]
    candidate = [dict(row, logits=[0, 2, 0, 0]) for row in baseline]
    result = compare_predictions(baseline, candidate)["candidate_minus_baseline_accuracy"]
    assert result["candidate_minus_baseline"] == -1.0
    assert "baseline_minus_candidate" not in result


def test_zero_gate_matches_baseline_has_gradients_and_preserves_option_order():
    torch = pytest.importorskip("torch")
    from scripts.egoviz_channel_scorer import ChannelWeightedScorer
    from scripts.train_egoviz_vs_sync_readout import JointTemporalScorer

    torch.manual_seed(17)
    baseline = JointTemporalScorer()
    torch.manual_seed(17)
    candidate = ChannelWeightedScorer()
    video = torch.randn(2, 8, 768)
    sensor = torch.randn(2, 4, 6, 16, 512)
    original = baseline(video, sensor)
    modified = candidate(video, sensor)
    torch.testing.assert_close(original, modified, atol=1e-6, rtol=1e-5)
    modified[:, 0].sum().backward()
    gradient = candidate.channel_logits.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient).item() > 0
    permutation = torch.tensor([2, 0, 3, 1])
    torch.testing.assert_close(candidate(video, sensor[:, permutation]), modified[:, permutation])
    assert sum(p.numel() for p in candidate.parameters()) == 84992
