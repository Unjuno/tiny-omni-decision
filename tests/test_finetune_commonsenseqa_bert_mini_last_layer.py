from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
MODULE_PATH = SCRIPT_DIR / "finetune_commonsenseqa_bert_mini_last_layer.py"
SPEC = importlib.util.spec_from_file_location("commonsenseqa_last_layer", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class _FakeEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = torch.nn.ModuleList([torch.nn.Linear(4, 4) for _ in range(4)])


class _FakeBert(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embeddings = torch.nn.Linear(4, 4)
        self.encoder = _FakeEncoder()


def test_only_final_transformer_layer_is_trainable() -> None:
    model = _FakeBert()

    trainable = MODULE.configure_last_layer_trainable(model)

    assert trainable == sum(parameter.numel() for parameter in model.encoder.layer[-1].parameters())
    assert all(not parameter.requires_grad for parameter in model.embeddings.parameters())
    assert all(
        not parameter.requires_grad
        for layer in model.encoder.layer[:-1]
        for parameter in layer.parameters()
    )
    assert all(parameter.requires_grad for parameter in model.encoder.layer[-1].parameters())


def test_experiment_has_a_fixed_three_epoch_budget_and_cpu_only_contract() -> None:
    assert MODULE.DEFAULT_EPOCHS == 3
    assert MODULE.ENCODER_LR == 2e-5
    assert MODULE.SCORER_LR == 1e-3
    assert MODULE.WEIGHT_DECAY == 0.01
