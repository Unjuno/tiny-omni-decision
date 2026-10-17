"""Text-free fixed-taxonomy audio Decision readout over frozen audio features."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .classification_json import classification_json


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
    def decide(self, audio_features: Tensor, options: list[dict[str, str]]) -> dict:
        """Return probability JSON for a supplied subset/order of known class IDs."""
        if audio_features.ndim != 1:
            raise ValueError("decide expects one audio feature vector")
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
        probabilities = torch.softmax(all_logits[indexes].float(), dim=-1).tolist()
        return classification_json(probabilities, requested, options)
