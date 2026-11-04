"""Input and metric helpers for one QASC text-criterion fusion diagnostic."""

from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F


def qasc_text_roles(row: dict[str, Any]) -> tuple[str, str, list[str]]:
    """Return state facts, natural-language criterion, and supplied options."""
    if row.get("split") != "train":
        raise ValueError("QASC criterion fusion accepts TRAIN rows only")
    state = row.get("state")
    question = row.get("question")
    options = row.get("options")
    target = row.get("target")
    if not isinstance(state, str) or not state.strip():
        raise ValueError("QASC state must contain the supplied support facts")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("QASC question must be a nonempty text criterion")
    if not isinstance(options, list) or len(options) != 8 or any(
        not isinstance(option, str) or not option.strip() for option in options
    ):
        raise ValueError("QASC needs eight nonempty supplied options")
    if len(set(options)) != 8 or not isinstance(target, str) or target not in options:
        raise ValueError("QASC options must be distinct and include the target")
    return state, question, options


def qasc_probability_metrics(
    probabilities: torch.Tensor, targets: torch.Tensor, *, bins: int = 15
) -> dict[str, float | int]:
    probabilities = probabilities.float()
    targets = targets.long()
    if probabilities.ndim != 2 or probabilities.shape != (len(targets), 8):
        raise ValueError("QASC probabilities must have shape [rows,8]")
    if not torch.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("probabilities must be finite and nonnegative")
    if not torch.allclose(probabilities.sum(-1), torch.ones(len(targets)), atol=1e-5):
        raise ValueError("probabilities must sum to one")
    correct = probabilities.argmax(-1).eq(targets)
    nll = -probabilities.gather(1, targets[:, None]).squeeze(1).clamp_min(1e-12).log()
    one_hot = F.one_hot(targets, num_classes=8).float()
    confidence = probabilities.max(-1).values
    ece = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        mask = (confidence >= low) & (
            confidence < high if index + 1 < bins else confidence <= high
        )
        if mask.any():
            ece += float(mask.float().mean()) * abs(
                float(confidence[mask].mean() - correct[mask].float().mean())
            )
    return {
        "examples": len(targets),
        "accuracy": float(correct.float().mean()),
        "nll": float(nll.mean()),
        "brier": float((probabilities - one_hot).square().sum(-1).mean()),
        "ece_15_bins": ece,
        "mean_confidence": float(confidence.mean()),
    }
