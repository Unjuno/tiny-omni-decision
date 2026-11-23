from __future__ import annotations

import torch

from scripts.pretrained_reuse.train_jaqket_frozen_readout import SharedOptionScorer, metrics


def test_shared_option_scorer_is_permutation_equivariant() -> None:
    torch.manual_seed(17)
    scorer = SharedOptionScorer(hidden_size=8).eval()
    features = torch.randn(3, 20, 8)
    permutation = torch.randperm(20)

    original = scorer(features)
    reordered = scorer(features[:, permutation])

    assert torch.allclose(reordered, original[:, permutation])


def test_choice_metrics_are_finite_and_chance_nll_is_log_twenty() -> None:
    logits = torch.zeros(4, 20)
    targets = torch.tensor([0, 1, 2, 3])

    result = metrics(logits, targets)

    assert result["accuracy"] == 0.25
    assert abs(result["nll"] - torch.log(torch.tensor(20.0)).item()) < 1e-6
    assert abs(result["brier"] - 0.95) < 1e-6
    assert abs(result["ece_15"] - 0.2) < 1e-6
