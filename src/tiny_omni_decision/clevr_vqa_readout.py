"""Frozen-feature CLEVR decision head and supplied-option objective."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from tiny_omni_decision.clevr_vqa_options import ANSWER_LABELS

FEATURE_WIDTH = 768
MAX_OPTIONS = 4


class ClevrAnswerReadout(nn.Module):
    """Small trainable answer head over a frozen joint image/question representation."""

    def __init__(self) -> None:
        super().__init__()
        self.classifier = nn.Linear(FEATURE_WIDTH, len(ANSWER_LABELS))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != FEATURE_WIDTH:
            raise ValueError(f"expected [batch, {FEATURE_WIDTH}] pooled features")
        return self.classifier(features)


def masked_option_logits(
    answer_logits: torch.Tensor,
    candidate_indices: torch.Tensor,
    option_counts: torch.Tensor,
) -> torch.Tensor:
    """Gather only supplied answer logits; padded candidate slots become negative infinity."""
    if answer_logits.ndim != 2 or answer_logits.shape[1] != len(ANSWER_LABELS):
        raise ValueError(f"expected [batch, {len(ANSWER_LABELS)}] answer logits")
    if candidate_indices.ndim != 2 or candidate_indices.shape[1] != MAX_OPTIONS:
        raise ValueError(f"candidate indices must have shape [batch, {MAX_OPTIONS}]")
    if candidate_indices.shape[0] != answer_logits.shape[0]:
        raise ValueError("candidate and answer-logit batch sizes differ")
    if option_counts.shape != (answer_logits.shape[0],):
        raise ValueError("option counts must have shape [batch]")
    if not torch.all((option_counts >= 2) & (option_counts <= MAX_OPTIONS)):
        raise ValueError("each example must provide two to four options")
    valid = (
        torch.arange(MAX_OPTIONS, device=answer_logits.device)[None, :]
        < option_counts[:, None]
    )
    if torch.any(candidate_indices[valid] < 0) or torch.any(
        candidate_indices[valid] >= len(ANSWER_LABELS)
    ):
        raise ValueError("candidate index is outside the CLEVR answer vocabulary")
    if torch.any(candidate_indices[~valid] != -1):
        raise ValueError("unused candidate slots must be padded with -1")
    selected = answer_logits.gather(1, candidate_indices.clamp_min(0))
    return selected.masked_fill(~valid, -torch.inf)


def option_masked_cross_entropy(
    answer_logits: torch.Tensor,
    candidate_indices: torch.Tensor,
    option_counts: torch.Tensor,
    target_positions: torch.Tensor,
) -> torch.Tensor:
    """Train with cross-entropy normalized over supplied candidates only."""
    if target_positions.shape != (answer_logits.shape[0],):
        raise ValueError("target positions must have shape [batch]")
    if torch.any(target_positions < 0) or torch.any(target_positions >= option_counts):
        raise ValueError("target position is outside the supplied options")
    choices = masked_option_logits(answer_logits, candidate_indices, option_counts)
    return F.cross_entropy(choices, target_positions)


def option_probabilities_batch(
    answer_logits: torch.Tensor,
    candidate_indices: torch.Tensor,
    option_counts: torch.Tensor,
) -> torch.Tensor:
    """Return supplied-option probabilities in the input option order."""
    choices = masked_option_logits(answer_logits, candidate_indices, option_counts)
    return torch.softmax(choices, dim=-1)


def quality_metrics(
    probabilities: torch.Tensor,
    target_positions: torch.Tensor,
    *,
    calibration_bins: int = 15,
) -> dict[str, float | int]:
    """Compute multiclass decision metrics for already option-masked probabilities."""
    if probabilities.ndim != 2 or probabilities.shape[0] == 0:
        raise ValueError("probabilities must be a non-empty [examples, options] tensor")
    if target_positions.shape != (probabilities.shape[0],):
        raise ValueError("target positions must have shape [examples]")
    if calibration_bins <= 0:
        raise ValueError("calibration_bins must be positive")
    if not torch.isfinite(probabilities).all() or torch.any(probabilities < 0):
        raise FloatingPointError("probabilities must be finite and non-negative")
    if not torch.allclose(
        probabilities.sum(dim=-1),
        torch.ones(probabilities.shape[0], device=probabilities.device, dtype=probabilities.dtype),
        atol=1e-5,
        rtol=1e-5,
    ):
        raise ValueError("each probability row must sum to one")
    if torch.any(target_positions < 0) or torch.any(target_positions >= probabilities.shape[1]):
        raise ValueError("target position is outside the probability row")

    prediction = probabilities.argmax(dim=-1)
    confidence = probabilities.max(dim=-1).values
    correct = prediction.eq(target_positions)
    one_hot = F.one_hot(target_positions, num_classes=probabilities.shape[1]).to(
        probabilities.dtype
    )
    nll = -probabilities.gather(1, target_positions[:, None]).clamp_min(1e-12).log().mean()
    brier = (probabilities - one_hot).square().sum(dim=-1).mean()
    ece = torch.zeros((), dtype=probabilities.dtype, device=probabilities.device)
    bin_index = torch.clamp((confidence * calibration_bins).long(), max=calibration_bins - 1)
    for index in range(calibration_bins):
        selected = bin_index == index
        if torch.any(selected):
            ece += selected.float().mean() * (
                confidence[selected].mean() - correct[selected].to(probabilities.dtype).mean()
            ).abs()
    return {
        "accuracy": float(correct.float().mean().item()),
        "nll": float(nll.item()),
        "brier": float(brier.item()),
        "ece": float(ece.item()),
        "mean_confidence": float(confidence.mean().item()),
        "count": int(probabilities.shape[0]),
    }


def macro_by_kind(
    per_kind_metrics: dict[str, dict[str, float | int]],
) -> dict[str, float | int]:
    """Macro-average already aggregated per-kind metrics, not sample counts."""
    if not per_kind_metrics:
        raise ValueError("at least one answer kind is required")
    result: dict[str, float | int] = {}
    for metric in ("accuracy", "nll", "brier", "ece"):
        values = [float(values[metric]) for values in per_kind_metrics.values()]
        result[metric] = sum(values) / len(values)
    result["minimum_kind_accuracy"] = min(
        float(values["accuracy"]) for values in per_kind_metrics.values()
    )
    result["kind_count"] = len(per_kind_metrics)
    return result
