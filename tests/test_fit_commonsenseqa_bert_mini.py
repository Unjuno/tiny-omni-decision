from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("safetensors")

import torch

MODULE_PATH = Path(__file__).parents[1] / "scripts" / "fit_commonsenseqa_bert_mini.py"
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("commonsenseqa_bert_mini_probe", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_alias_soft_target_nll_matches_uniform_mass_on_equivalent_choices() -> None:
    logits = torch.tensor([[0.0, 0.0, 1.0, -1.0, -2.0]])
    targets = torch.tensor([[0.5, 0.5, 0.0, 0.0, 0.0]])

    nll = MODULE.soft_target_nll(logits, targets)
    expected = -torch.log_softmax(logits, dim=-1)[0, :2].mean()

    assert torch.allclose(nll, expected)


def test_validation_metrics_report_sample_and_concept_macro_scores() -> None:
    logits = torch.tensor(
        [
            [3.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 0.0, 0.0],
        ]
    )
    targets = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
        ]
    )

    metrics = MODULE.evaluate_logits(logits, targets, ["one", "one", "two"])

    assert metrics["accuracy"] == 1.0
    assert metrics["macro_concept_accuracy"] == 1.0
    assert metrics["concept_groups"] == 2
    assert metrics["nll"] > 0
    assert 0 <= metrics["ece_10"] <= 1


def test_shared_scalar_scorer_is_permutation_equivariant() -> None:
    torch.manual_seed(17)
    scorer = torch.nn.Linear(256, 1)
    features = torch.randn(7, 5, 256)
    permutation = torch.tensor([4, 1, 3, 0, 2])

    original = scorer(features).squeeze(-1)
    permuted = scorer(features[:, permutation]).squeeze(-1)

    assert torch.equal(permuted, original[:, permutation])
