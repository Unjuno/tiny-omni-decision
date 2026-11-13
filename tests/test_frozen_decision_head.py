from __future__ import annotations

import pytest

pytest.importorskip("torch")

import torch

from tiny_omni_decision.decision import FrozenFeatureCandidateScorer


def test_frozen_feature_head_emits_calibrated_option_distribution() -> None:
    head = FrozenFeatureCandidateScorer(feature_size=4)
    features = torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]])

    result = head.decide(features, temperature=0.5)

    assert result.option_logits.shape == (2,)
    assert result.probabilities.shape == (2,)
    assert result.probabilities.sum().item() == pytest.approx(1.0)
    assert result.predicted_index == int(result.probabilities.argmax().item())
    assert result.confidence == pytest.approx(float(result.probabilities.max().item()))


def test_frozen_feature_head_is_candidate_order_equivariant() -> None:
    torch.manual_seed(17)
    head = FrozenFeatureCandidateScorer(feature_size=6).eval()
    features = torch.randn(4, 6)
    permutation = torch.tensor([2, 0, 3, 1])

    original = head.decide(features)
    reordered = head.decide(features[permutation])

    torch.testing.assert_close(reordered.option_logits, original.option_logits[permutation])
    torch.testing.assert_close(reordered.probabilities, original.probabilities[permutation])


def test_frozen_feature_head_validates_candidate_shape_and_temperature() -> None:
    head = FrozenFeatureCandidateScorer(feature_size=3)
    with pytest.raises(ValueError, match="at least two"):
        head.decide(torch.zeros(1, 3))
    with pytest.raises(ValueError, match="feature size"):
        head.decide(torch.zeros(2, 4))
    with pytest.raises(ValueError, match="temperature"):
        head.decide(torch.zeros(2, 3), temperature=0.0)
