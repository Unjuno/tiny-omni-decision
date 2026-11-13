from __future__ import annotations

import torch

from tiny_omni_decision.cater_temporal_probe import CaterTemporalDecisionHead


def test_temporal_decision_head_scores_each_option_and_backpropagates() -> None:
    torch.manual_seed(17)
    model = CaterTemporalDecisionHead(hidden_dim=16)
    video = torch.randn(13, 256, 768)
    question = torch.randn(384)
    options = torch.randn(7, 384)

    logits = model(video, question, options, 0, 3)
    logits[2].backward()

    assert logits.shape == (7,)
    assert torch.isfinite(logits).all()
    assert model.key_projection.weight.grad is not None
    assert model.temporal_reader.weight_ih_l0.grad is not None


def test_temporal_decision_head_is_candidate_order_equivariant() -> None:
    torch.manual_seed(23)
    model = CaterTemporalDecisionHead(hidden_dim=16).eval()
    video = torch.randn(13, 256, 768)
    question = torch.randn(384)
    options = torch.randn(5, 384)
    permutation = torch.tensor([3, 0, 4, 1, 2])

    original = model(video, question, options, 3, 6)
    permuted = model(video, question, options[permutation], 3, 6)

    assert torch.allclose(permuted, original[permutation], atol=1e-6)
