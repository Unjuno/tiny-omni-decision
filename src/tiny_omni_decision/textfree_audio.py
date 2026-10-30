"""Text-free fixed-taxonomy audio Decision readout over frozen audio features."""

from __future__ import annotations

import hashlib
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .classification_json import classification_json


def speaker_group_split(
    speaker_group_ids: list[str], *, seed: int = 17, calibration_fraction: float = 0.2
) -> tuple[list[int], list[int]]:
    """Split row indices by speaker group using a stable hash ordering."""
    if (
        not speaker_group_ids
        or any(not isinstance(value, str) or not value for value in speaker_group_ids)
        or not math.isfinite(calibration_fraction)
        or not 0 < calibration_fraction < 1
    ):
        raise ValueError("invalid speaker groups or calibration fraction")
    groups = sorted(set(speaker_group_ids))
    if len(groups) < 2:
        raise ValueError("at least two speaker groups are required")
    ranked = sorted(
        groups,
        key=lambda group: hashlib.sha256(
            f"{seed}:calibration:{group}".encode()
        ).hexdigest(),
    )
    calibration_groups = set(ranked[: math.ceil(len(ranked) * calibration_fraction)])
    calibration = [
        index for index, group in enumerate(speaker_group_ids) if group in calibration_groups
    ]
    fit = [
        index for index, group in enumerate(speaker_group_ids) if group not in calibration_groups
    ]
    return fit, calibration


class TextFreeAudioReadout(nn.Module):
    """Map one frozen audio vector directly to a fixed set of action classes."""

    def __init__(self, feature_width: int, class_ids: list[str]):
        super().__init__()
        if (
            feature_width < 1
            or len(class_ids) < 2
            or any(not isinstance(item, str) or not item.strip() for item in class_ids)
            or len(set(class_ids)) != len(class_ids)
        ):
            raise ValueError("class IDs must be nonempty strings")
        self.class_ids = tuple(class_ids)
        self.classifier = nn.Linear(feature_width, len(class_ids))

    def forward(self, audio_features: Tensor) -> Tensor:
        if audio_features.ndim not in (1, 2):
            raise ValueError("audio features must have shape [width] or [batch, width]")
        if audio_features.shape[-1] != self.classifier.in_features:
            raise ValueError("audio feature width does not match the readout")
        if not torch.isfinite(audio_features).all():
            raise ValueError("audio features must be finite")
        return self.classifier(audio_features)

    @torch.inference_mode()
    def decide(
        self,
        audio_features: Tensor,
        options: list[dict[str, str]],
        temperature: float = 1.0,
    ) -> dict:
        """Return probability JSON for a supplied subset/order of known class IDs."""
        if audio_features.ndim != 1:
            raise ValueError("decide expects one audio feature vector")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if any(
            not isinstance(option, dict) or set(option) != {"option_id", "class_id"}
            for option in options
        ):
            raise ValueError("options must contain only option_id and class_id")
        requested = [option.get("class_id") for option in options]
        if (
            len(requested) < 2
            or any(item not in self.class_ids for item in requested)
            or len(set(requested)) != len(requested)
        ):
            raise ValueError("options must contain at least two unique known class IDs")
        all_logits = self(audio_features)
        indexes = torch.tensor([self.class_ids.index(item) for item in requested])
        probabilities = torch.softmax(all_logits[indexes].float() / temperature, dim=-1).tolist()
        return classification_json(probabilities, requested, options)


def fit_scalar_temperature(
    calibration_logits: Tensor,
    targets: Tensor,
    *,
    max_iter: int = 100,
) -> float:
    """Fit one positive temperature from a separate calibration split only."""
    if (
        calibration_logits.ndim != 2
        or calibration_logits.shape[0] < 2
        or calibration_logits.shape[1] < 2
        or targets.ndim != 1
        or targets.shape[0] != calibration_logits.shape[0]
        or targets.dtype not in (torch.int32, torch.int64)
        or max_iter < 1
    ):
        raise ValueError("invalid calibration logits, targets, or iteration limit")
    if not torch.isfinite(calibration_logits).all():
        raise ValueError("calibration logits must be finite")
    if targets.min().item() < 0 or targets.max().item() >= calibration_logits.shape[1]:
        raise ValueError("calibration target is outside the logit class range")

    logits = calibration_logits.detach().to(dtype=torch.float64)
    labels = targets.detach().to(dtype=torch.long)
    log_temperature = nn.Parameter(torch.zeros((), dtype=torch.float64))
    optimizer = torch.optim.LBFGS(
        [log_temperature],
        lr=0.1,
        max_iter=max_iter,
        tolerance_grad=1e-10,
        tolerance_change=1e-12,
        line_search_fn="strong_wolfe",
    )

    def closure() -> Tensor:
        optimizer.zero_grad(set_to_none=True)
        temperature = log_temperature.exp().clamp(0.05, 20.0)
        loss = F.cross_entropy(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(0.05, 20.0).item())
