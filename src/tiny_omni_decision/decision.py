from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass
class DecisionOutput:
    option_logits: Tensor
    probabilities: Tensor
    predicted_index: int
    confidence: float


def option_logits_from_vocab(last_logits: Tensor, label_ids: list[int]) -> Tensor:
    if last_logits.ndim != 1:
        raise ValueError("last_logits must be a one-dimensional vocabulary-logit vector")
    return last_logits[label_ids]


def normalize_probabilities(logits: Tensor) -> Tensor:
    if logits.ndim != 1 or logits.numel() < 2:
        raise ValueError("option logits must be a vector with at least two options")
    return torch.softmax(logits.float(), dim=-1)


def brier_loss(probabilities: Tensor, target_indices: Tensor) -> Tensor:
    if probabilities.ndim != 2:
        raise ValueError("probabilities must have shape [batch, options]")
    targets = F.one_hot(target_indices, num_classes=probabilities.shape[-1]).to(probabilities.dtype)
    return ((probabilities - targets) ** 2).sum(dim=-1).mean()


def decision_loss(
    logits: Tensor,
    target_indices: Tensor,
    brier_weight: float = 0.2,
    cross_entropy_weight: float = 1.0,
) -> tuple[Tensor, Tensor, Tensor]:
    probabilities = torch.softmax(logits.float(), dim=-1)
    ce = F.cross_entropy(logits.float(), target_indices)
    brier = brier_loss(probabilities, target_indices)
    return cross_entropy_weight * ce + brier_weight * brier, ce, brier
