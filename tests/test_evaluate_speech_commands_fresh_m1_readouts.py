"""CPU-only checks for fresh-speaker paired evaluation accounting."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

MODULE_PATH = (
    Path(__file__).parents[1] / "scripts" / "evaluate_speech_commands_fresh_m1_readouts.py"
)
SPEC = importlib.util.spec_from_file_location("fresh_m1_eval", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _group_stats(probabilities: torch.Tensor, targets: torch.Tensor, groups: list[str]):
    result = MODULE._group_stats(probabilities, targets, groups)
    return result[1:]


def test_group_stats_preserve_grouped_example_accounting() -> None:
    probabilities = torch.zeros((4, len(MODULE.LABELS)))
    probabilities[0, 0] = 0.9
    probabilities[1, 1] = 0.8
    probabilities[2, 2] = 0.7
    probabilities[3, 3] = 0.6
    targets = torch.tensor([0, 0, 2, 3])
    names, counts, correct, confidence, totals, sizes = MODULE._group_stats(
        probabilities, targets, ["speaker-b", "speaker-a", "speaker-b", "speaker-c"]
    )

    assert names == ["speaker-a", "speaker-b", "speaker-c"]
    assert sizes.tolist() == [1.0, 2.0, 1.0]
    assert counts.sum(axis=1).tolist() == [1.0, 2.0, 1.0]
    assert correct.sum() == 3.0
    assert confidence.sum() == pytest.approx(3.0)
    assert totals[:, 0].sum() == 3.0


def test_paired_speaker_bootstrap_is_deterministic_and_detects_metric_tradeoff() -> None:
    targets = torch.tensor([0, 1, 2, 3, 4, 5])
    baseline = torch.full((6, len(MODULE.LABELS)), 0.01)
    candidate = baseline.clone()
    for row, target in enumerate(targets.tolist()):
        baseline[row, target] = 0.91
        candidate[row, target] = 0.94
    baseline /= baseline.sum(dim=1, keepdim=True)
    candidate /= candidate.sum(dim=1, keepdim=True)
    groups = ["speaker-1", "speaker-1", "speaker-2", "speaker-2", "speaker-3", "speaker-3"]
    base_stats = _group_stats(baseline, targets, groups)
    cand_stats = _group_stats(candidate, targets, groups)

    first = MODULE._bootstrap(base_stats, cand_stats, seed=17, repeats=1000)
    second = MODULE._bootstrap(base_stats, cand_stats, seed=17, repeats=1000)

    assert first == second
    assert first["groups"] == 3
    assert np.allclose(first["candidate_minus_baseline_ci95"]["accuracy"], [0.0, 0.0])
    assert first["candidate_minus_baseline_ci95"]["nll"][1] < 0


def test_paired_bootstrap_rejects_unequal_group_counts() -> None:
    targets = torch.tensor([0, 1, 2])
    probabilities = torch.full((3, len(MODULE.LABELS)), 0.01)
    probabilities[range(3), targets] = 0.91
    probabilities /= probabilities.sum(dim=1, keepdim=True)
    two_groups = _group_stats(probabilities, targets, ["a", "a", "b"])
    three_groups = _group_stats(probabilities, targets, ["a", "b", "c"])

    with pytest.raises(ValueError, match="same speaker groups"):
        MODULE._bootstrap(two_groups, three_groups, repeats=1000)
