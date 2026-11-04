from __future__ import annotations

import pytest

from tiny_omni_decision.qasc_text_criterion_fusion import (
    qasc_probability_metrics,
    qasc_text_roles,
)


def _row(split="train"):
    return {
        "split": split,
        "state": "Evidence: fact one\nFact two.",
        "question": "Which answer follows?",
        "options": [f"option-{index}" for index in range(8)],
        "target": "option-3",
    }


def test_qasc_roles_keep_state_criterion_and_options_separate():
    state, criterion, options = qasc_text_roles(_row())

    assert state == "Evidence: fact one\nFact two."
    assert criterion == "Which answer follows?"
    assert options == [f"option-{index}" for index in range(8)]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda row: row.update(split="validation"),
        lambda row: row.update(question=" "),
        lambda row: row.update(options=["same"] * 8),
        lambda row: row.update(target="not-supplied"),
    ],
)
def test_qasc_roles_reject_invalid_or_nontrain_rows(mutate):
    row = _row()
    mutate(row)

    with pytest.raises(ValueError):
        qasc_text_roles(row)


def test_qasc_probability_metrics_report_calibrated_uniform_control():
    torch = pytest.importorskip("torch")
    probabilities = torch.full((4, 8), 1 / 8)
    targets = torch.tensor([0, 1, 2, 3])

    metrics = qasc_probability_metrics(probabilities, targets)

    assert metrics["examples"] == 4
    assert metrics["accuracy"] == pytest.approx(0.25)
    assert metrics["nll"] == pytest.approx(torch.log(torch.tensor(8.0)).item())
    assert metrics["brier"] == pytest.approx(0.875)
    assert metrics["mean_confidence"] == pytest.approx(0.125)


def test_qasc_probability_metrics_reject_invalid_rows():
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError, match="sum to one"):
        qasc_probability_metrics(torch.full((2, 8), 0.2), torch.tensor([0, 1]))
