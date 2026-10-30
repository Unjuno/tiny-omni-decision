"""Text-free Decision readout over one frozen modality feature and typed options.

This module deliberately accepts numeric feature tensors only. Raw-media encoding,
option-attribute encoding, evidence freshness, and modality-specific abstention stay
in the caller; importing this module never loads a language encoder.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .classification_json import classification_json
from .decision import FrozenFeatureCandidateScorer


class NativeFeatureDecisionScorer(nn.Module):
    """Candidate-equivariant scorer for one modality and explicit option vectors.

    ``observation_features`` has shape ``[batch, feature_width]`` and contains a
    frozen feature from exactly one modality. ``option_features`` has shape
    ``[batch, candidates, option_width]`` and must be produced from a fixed,
    task-owned typed schema (for example numeric action fields or enum codes),
    not from option names or a hidden text question.
    """

    def __init__(self, feature_width: int, option_width: int, hidden_width: int = 128):
        super().__init__()
        if min(feature_width, option_width, hidden_width) < 1:
            raise ValueError("feature, option, and hidden widths must be positive")
        self.feature_width = feature_width
        self.option_width = option_width
        self.scorer = FrozenFeatureCandidateScorer(
            feature_width + option_width, hidden_size=hidden_width
        )

    def forward(
        self,
        observation_features: Tensor,
        option_features: Tensor,
        option_valid: Tensor,
    ) -> Tensor:
        if (observation_features.ndim != 2
                or observation_features.shape[-1] != self.feature_width):
            raise ValueError("observation features must have shape [batch, feature_width]")
        if (option_features.ndim != 3
                or option_features.shape[0] != observation_features.shape[0]
                or option_features.shape[-1] != self.option_width):
            raise ValueError("option features must have shape [batch, candidates, option_width]")
        if (option_valid.dtype != torch.bool
                or option_valid.shape != option_features.shape[:2]
                or not (option_valid.sum(-1) >= 2).all()):
            raise ValueError("a boolean mask with at least two options per row is required")
        if (not torch.isfinite(observation_features).all()
                or not torch.isfinite(option_features[option_valid]).all()):
            raise ValueError("available observation and option features must be finite")

        clean_options = torch.where(
            option_valid[..., None], option_features, torch.zeros_like(option_features)
        )
        state = observation_features[:, None, :].expand(-1, option_features.shape[1], -1)
        logits = self.scorer(torch.cat((state, clean_options), dim=-1))
        return logits.masked_fill(~option_valid, float("-inf"))

    @torch.inference_mode()
    def decide(
        self,
        observation_features: Tensor,
        option_features: Tensor,
        option_ids: list[str],
        class_ids: list[str],
        *,
        temperature: float = 1.0,
    ) -> dict[str, object]:
        """Score a complete supplied typed-option set and return option-ID JSON."""
        if observation_features.ndim != 1 or option_features.ndim != 2:
            raise ValueError("decide expects one observation and an unbatched option matrix")
        if (len(option_ids) != len(option_features)
                or len(class_ids) != len(option_features)
                or len(set(option_ids)) != len(option_ids)
                or len(set(class_ids)) != len(class_ids)
                or any(not isinstance(value, str) or not value.strip()
                       for value in [*option_ids, *class_ids])):
            raise ValueError("unique nonempty option and class IDs must match the option rows")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")

        logits = self(
            observation_features[None],
            option_features[None],
            torch.ones((1, len(option_features)), dtype=torch.bool,
                       device=option_features.device),
        )[0].float() / temperature
        probabilities = torch.softmax(logits, dim=-1).cpu().tolist()
        options = [
            {"option_id": option_id, "class_id": class_id}
            for option_id, class_id in zip(option_ids, class_ids, strict=True)
        ]
        return classification_json(probabilities, class_ids, options)
