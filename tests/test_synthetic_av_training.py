import torch

from tiny_omni_decision.synthetic_av_training import (
    BestByNLL,
    classification_metrics,
    epoch_permutations,
)


def test_classification_metrics_report_binary_accuracy_nll_brier_ece():
    logits = torch.tensor([[4.0, 0.0], [4.0, 0.0]])
    targets = torch.tensor([0, 1])
    result = classification_metrics(logits, targets, ece_bins=10)
    assert result["accuracy"] == 0.5
    assert result["nll"] > 0.5
    assert 0.0 <= result["brier"] <= 2.0
    assert 0.0 <= result["ece"] <= 1.0
    assert result["mean_confidence"] > 0.5


def test_checkpoint_selector_uses_lowest_nll_and_keeps_earliest_tie():
    selector = BestByNLL()
    assert selector.update(0.8, 512)
    assert not selector.update(0.8, 1024)
    assert selector.step == 512
    assert selector.update(0.7, 1536)
    assert selector.step == 1536


def test_epoch_permutations_are_deterministic_and_cover_each_sample_once():
    first = epoch_permutations(64, epochs=5, seed=17)
    assert first == epoch_permutations(64, epochs=5, seed=17)
    assert all(sorted(epoch) == list(range(64)) for epoch in first)
