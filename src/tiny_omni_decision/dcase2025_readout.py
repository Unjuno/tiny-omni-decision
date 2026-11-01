"""Small text-free Decision heads for the frozen DCASE AV feature probe."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class DCASE2025Readout(nn.Module):
    """Score the fixed ``offscreen`` / ``onscreen`` options for an event class."""

    MODES = {"audio", "video", "joint"}

    def __init__(self, mode: str, *, hidden_width: int = 64, event_classes: int = 13):
        super().__init__()
        if mode not in self.MODES or min(hidden_width, event_classes) < 1:
            raise ValueError("unsupported mode or non-positive readout width")
        self.mode = mode
        self.event_classes = event_classes
        input_width = 13 + (768 if mode in {"audio", "joint"} else 0)
        input_width += 768 if mode in {"video", "joint"} else 0
        self.network = nn.Sequential(
            nn.Linear(input_width, hidden_width),
            nn.GELU(),
            nn.Linear(hidden_width, 2),
        )

    def forward(self, audio: Tensor, video: Tensor, event_class: Tensor) -> Tensor:
        if event_class.ndim != 1 or event_class.dtype != torch.long:
            raise ValueError("event_class must be a one-dimensional int64 tensor")
        batch = event_class.shape[0]
        if batch < 1 or (event_class < 0).any() or (event_class >= self.event_classes).any():
            raise ValueError("event_class is outside the fixed taxonomy")
        features = []
        if self.mode in {"audio", "joint"}:
            if audio.shape != (batch, 768) or not torch.isfinite(audio).all():
                raise ValueError("audio features must be finite [batch, 768]")
            features.append(audio)
        if self.mode in {"video", "joint"}:
            if video.shape != (batch, 768) or not torch.isfinite(video).all():
                raise ValueError("video features must be finite [batch, 768]")
            features.append(video)
        if not features:
            raise ValueError("at least one modality is required")
        context = torch.nn.functional.one_hot(event_class, self.event_classes).to(
            dtype=features[0].dtype
        )
        return self.network(torch.cat([*features, context], dim=-1))


def binary_probability_metrics(probabilities, targets, *, bins: int = 15) -> dict[str, float]:
    """Return fixed-order binary metrics; probabilities are ``[off, on]``."""
    probabilities = torch.as_tensor(probabilities, dtype=torch.float64)
    targets = torch.as_tensor(targets, dtype=torch.long)
    if (
        probabilities.ndim != 2
        or probabilities.shape[1] != 2
        or targets.shape != (len(probabilities),)
        or len(targets) == 0
        or bins < 1
        or not torch.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or (probabilities > 1).any()
        or not torch.allclose(probabilities.sum(1), torch.ones(len(targets), dtype=torch.float64))
        or ((targets != 0) & (targets != 1)).any()
    ):
        raise ValueError("expected finite normalized [off, on] probabilities and binary targets")
    prediction = probabilities.argmax(1)
    accuracy = float((prediction == targets).double().mean())
    per_class = []
    for label in (0, 1):
        mask = targets == label
        per_class.append(
            float((prediction[mask] == label).double().mean()) if mask.any() else math.nan
        )
    nll = float(
        -torch.log(probabilities[torch.arange(len(targets)), targets].clamp_min(1e-15)).mean()
    )
    one_hot = torch.nn.functional.one_hot(targets, 2).double()
    brier = float(((probabilities - one_hot) ** 2).sum(1).mean())
    confidence = probabilities.max(1).values
    ece = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        mask = (confidence >= lower) & (
            confidence <= upper if index == bins - 1 else confidence < upper
        )
        if mask.any():
            ece += float(mask.double().mean()) * abs(
                float((prediction[mask] == targets[mask]).double().mean())
                - float(confidence[mask].mean())
            )
    return {
        "count": len(targets),
        "accuracy": accuracy,
        "balanced_accuracy": sum(per_class) / 2
        if all(math.isfinite(x) for x in per_class)
        else math.nan,
        "recall_offscreen": per_class[0],
        "recall_onscreen": per_class[1],
        "nll": nll,
        "brier": brier,
        "ece_15": ece,
        "mean_confidence": float(confidence.mean()),
    }
