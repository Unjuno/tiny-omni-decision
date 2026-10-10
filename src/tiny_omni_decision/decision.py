from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class DecisionOutput:
    option_logits: Tensor
    probabilities: Tensor
    predicted_index: int
    confidence: float


class FrozenFeatureCandidateScorer(nn.Module):
    """Score each supplied option from precomputed observation/question features.

    Encoders and feature construction stay outside this module, so callers can
    reuse one cached observation representation across independently authored
    questions and variable-sized candidate sets.
    """

    def __init__(self, feature_size: int, hidden_size: int = 128) -> None:
        super().__init__()
        if feature_size < 1 or hidden_size < 1:
            raise ValueError("feature_size and hidden_size must be positive")
        self.feature_size = feature_size
        self.network = nn.Sequential(
            nn.Linear(feature_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, candidate_features: Tensor) -> Tensor:
        if candidate_features.ndim < 2:
            raise ValueError("candidate features must include option and feature dimensions")
        if candidate_features.shape[-1] != self.feature_size:
            raise ValueError(
                f"candidate feature size must be {self.feature_size}, "
                f"got {candidate_features.shape[-1]}"
            )
        return self.network(candidate_features).squeeze(-1)

    def decide(self, candidate_features: Tensor, temperature: float = 1.0) -> DecisionOutput:
        if candidate_features.ndim != 2 or candidate_features.shape[0] < 2:
            raise ValueError("candidate features must contain at least two options")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        logits = self(candidate_features).float() / temperature
        if not torch.isfinite(logits).all():
            raise ValueError("candidate scoring produced non-finite option logits")
        probabilities = normalize_probabilities(logits)
        predicted_index = int(probabilities.argmax().item())
        return DecisionOutput(
            option_logits=logits,
            probabilities=probabilities,
            predicted_index=predicted_index,
            confidence=float(probabilities[predicted_index].item()),
        )


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
