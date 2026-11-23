from __future__ import annotations

import torch

from scripts.pretrained_reuse.run_socialiqa_minilm_probe import (
    OptionScorer,
    grouped_split,
    metrics,
)


def test_grouped_split_is_deterministic_and_keeps_contexts_together() -> None:
    items = [{"group": f"context-{index // 2}", "target": index % 3} for index in range(30)]
    train_a, validation_a = grouped_split(items, seed=17, validation_fraction=0.2)
    train_b, validation_b = grouped_split(items, seed=17, validation_fraction=0.2)
    assert (train_a, validation_a) == (train_b, validation_b)
    assert set(train_a).isdisjoint(validation_a)
    assert set(train_a) | set(validation_a) == set(range(len(items)))
    train_groups = {items[index]["group"] for index in train_a}
    validation_groups = {items[index]["group"] for index in validation_a}
    assert train_groups.isdisjoint(validation_groups)


def test_shared_option_scorer_is_equivariant_to_candidate_order() -> None:
    torch.manual_seed(17)
    model = OptionScorer(8)
    features = torch.randn(4, 3, 8)
    permutation = torch.tensor([2, 0, 1])
    original = model(features)
    permuted = model(features[:, permutation])
    torch.testing.assert_close(permuted, original[:, permutation])


def test_metrics_are_finite_and_match_perfect_three_choice_predictions() -> None:
    logits = torch.tensor([[6.0, 0.0, -1.0], [0.0, 6.0, -1.0], [-1.0, 0.0, 6.0]])
    labels = torch.tensor([0, 1, 2])
    observed = metrics(logits, labels)
    assert observed["accuracy"] == 1.0
    assert observed["nll"] < 0.01
    assert observed["brier"] < 0.01
    assert observed["ece_10"] < 0.01
