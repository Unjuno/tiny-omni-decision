from __future__ import annotations

import torch

from scripts.train_gsm8k_crossencoder import epoch_option_order, grouped_choice_loss
from tiny_omni_decision.schema import DecisionExample, LicenseProvenance


def _example() -> DecisionExample:
    return DecisionExample(
        id="gsm8k-test-1",
        modality="text",
        state="",
        question="A short arithmetic question.",
        options=["10", "11", "12", "13"],
        target="12",
        source="openai/grade-school-math",
        source_revision="3101c7d5072418e28b9008a6636bde82a006892c",
        source_record_id="a" * 64,
        split="train",
        provenance=LicenseProvenance(license="MIT", trust_status="trusted"),
    )


def test_epoch_option_shuffle_is_deterministic_and_keeps_target_identity() -> None:
    example = _example()
    options_a, target_a = epoch_option_order(example, 1)
    options_b, target_b = epoch_option_order(example, 1)
    options_next, target_next = epoch_option_order(example, 2)

    assert options_a == options_b
    assert target_a == target_b
    assert options_a[target_a] == example.target
    assert set(options_a) == set(example.options)
    assert options_next[target_next] == example.target
    assert set(options_next) == set(example.options)


def test_grouped_choice_loss_is_finite_and_uses_one_target_per_question() -> None:
    scores = torch.tensor([[0.0, 2.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    targets = torch.tensor([1, 0])
    loss = grouped_choice_loss(scores, targets)
    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_grouped_choice_loss_rejects_non_four_choice_inputs() -> None:
    with torch.no_grad():
        scores = torch.zeros((2, 3))
        targets = torch.zeros((2,), dtype=torch.long)
    try:
        grouped_choice_loss(scores, targets)
    except ValueError as exc:
        assert "shape" in str(exc)
    else:
        raise AssertionError("expected four-choice validation")
