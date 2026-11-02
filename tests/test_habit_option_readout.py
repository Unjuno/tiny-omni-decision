import numpy as np
import pytest
import torch

from scripts.train_habit_option_readout import (
    MODES,
    OptionReadout,
    classification_metrics,
    cluster_bootstrap_delta,
)


def test_metrics_match_uniform_two_choice_baseline():
    metrics = classification_metrics(np.asarray([[0.5, 0.5], [0.5, 0.5]]), np.asarray([0, 1]))
    assert metrics["accuracy"] == 0.5
    assert metrics["nll"] == pytest.approx(np.log(2))
    assert metrics["brier"] == pytest.approx(0.5)
    assert metrics["ece_15"] == 0.0


@pytest.mark.parametrize("mode", MODES)
def test_readout_modes_mask_padding_and_backpropagate(mode):
    model = OptionReadout(mode, max_option_id=38)
    video = torch.randn(2, 8, 768)
    state = torch.randn(2, 8, 42)
    candidates = torch.randn(2, 4, 384)
    candidate_ids = torch.tensor([[3, 9, 12, -1], [2, 7, -1, -1]])
    mask = candidate_ids >= 0
    logits = model(video, state, candidates, candidate_ids, mask)
    assert logits.shape == (2, 4)
    assert torch.all(logits[~mask] < -1e8)
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([1, 0]))
    loss.backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_task_group_bootstrap_is_seeded_and_reports_group_limit():
    left = np.asarray([1, 1, 0, 0, 1, 0], dtype=bool)
    right = np.asarray([0, 1, 0, 0, 0, 0], dtype=bool)
    groups = np.asarray(["a", "a", "b", "b", "c", "c"])
    first = cluster_bootstrap_delta(left, right, groups, repetitions=500)
    second = cluster_bootstrap_delta(left, right, groups, repetitions=500)
    assert first == second
    assert first["unit"] == "task_group"
    assert len(first["groups"]) == 3
