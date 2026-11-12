"""Small permutation-equivariant scorer for frozen MiniWoB screen features."""

from __future__ import annotations

import math
import random

import torch
from torch import nn


class MiniWoBOptionScorer(nn.Module):
    """Score each candidate from shared screen context, local visual tokens and xy."""

    def __init__(self, feature_width: int = 1154, hidden_width: int = 64) -> None:
        super().__init__()
        if feature_width < 1 or hidden_width < 1:
            raise ValueError("scorer dimensions must be positive")
        self.feature_width = feature_width
        self.network = nn.Sequential(
            nn.LayerNorm(feature_width),
            nn.Linear(feature_width, hidden_width),
            nn.GELU(),
            nn.Linear(hidden_width, 1),
        )

    def forward(self, option_features: torch.Tensor) -> torch.Tensor:
        if (
            option_features.ndim != 3
            or option_features.shape[1] < 2
            or option_features.shape[2] != self.feature_width
            or not torch.isfinite(option_features).all()
        ):
            raise ValueError("finite [batch, options>=2, configured feature_width] required")
        return self.network(option_features).squeeze(-1)


def classification_metrics(probabilities: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    """Four-choice accuracy, NLL, multiclass Brier, top-label ECE and confidence."""
    if (
        probabilities.ndim != 2
        or targets.ndim != 1
        or probabilities.shape[0] != targets.shape[0]
        or probabilities.shape[0] < 1
        or probabilities.shape[1] < 2
        or not torch.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or (probabilities > 1).any()
        or not torch.allclose(
            probabilities.sum(dim=1),
            torch.ones_like(targets, dtype=probabilities.dtype),
            atol=1e-5,
            rtol=1e-5,
        )
        or (targets < 0).any()
        or (targets >= probabilities.shape[1]).any()
    ):
        raise ValueError("valid probability rows and in-range targets required")
    prediction = probabilities.argmax(dim=1)
    confidence = probabilities.max(dim=1).values
    correct = prediction.eq(targets)
    one_hot = torch.nn.functional.one_hot(targets, num_classes=probabilities.shape[1]).to(
        probabilities.dtype
    )
    ece = 0.0
    for bin_index in range(15):
        lower = bin_index / 15
        upper = (bin_index + 1) / 15
        in_bin = (confidence >= lower) & (
            confidence <= upper if bin_index == 14 else confidence < upper
        )
        if in_bin.any():
            ece += float(in_bin.float().mean()) * abs(
                float(correct[in_bin].float().mean()) - float(confidence[in_bin].mean())
            )
    return {
        "count": int(targets.shape[0]),
        "accuracy": float(correct.float().mean()),
        "nll": float(-probabilities.gather(1, targets[:, None]).clamp_min(1e-12).log().mean()),
        "brier": float((probabilities - one_hot).square().sum(dim=1).mean()),
        "ece_15_bins": ece,
        "mean_confidence": float(confidence.mean()),
    }


def permute_candidate_axis(
    features: torch.Tensor, permutation: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return reordered features and an inverse permutation for restoring scores."""
    if (
        features.ndim != 3
        or permutation.ndim != 1
        or permutation.numel() != features.shape[1]
        or sorted(permutation.tolist()) != list(range(features.shape[1]))
    ):
        raise ValueError("a complete option-axis permutation is required")
    inverse = torch.argsort(permutation)
    return features.index_select(1, permutation), inverse


def deranged_indices(count: int, seed: int) -> list[int]:
    """Build a deterministic full derangement using Sattolo's cycle shuffle."""
    if count < 2:
        raise ValueError("at least two items are required for a derangement")
    indices = list(range(count))
    rng = random.Random(seed)
    for index in range(count - 1, 0, -1):
        swap = rng.randrange(index)
        indices[index], indices[swap] = indices[swap], indices[index]
    if any(index == donor for index, donor in enumerate(indices)):
        raise RuntimeError("derangement construction produced a fixed point")
    return indices


def finite_json_metrics(metrics: dict[str, float]) -> dict[str, float | int]:
    """Reject non-finite metrics before they enter experiment artifacts."""
    result: dict[str, float | int] = {}
    for key, value in metrics.items():
        if key == "count":
            result[key] = int(value)
        elif math.isfinite(float(value)):
            result[key] = float(value)
        else:
            raise ValueError(f"metric {key} must be finite")
    return result
