"""Text-free image Decision readout over frozen image features."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn

from .classification_json import classification_json
from .native_decision_json import native_classification_request, native_decision_json


class TextFreeImageReadout(nn.Module):
    """Predict fixed image taxonomies from one frozen image feature vector."""

    def __init__(self, feature_width: int, taxonomies: Mapping[str, Sequence[str]]) -> None:
        super().__init__()
        if feature_width < 1 or not taxonomies:
            raise ValueError("feature width and taxonomies must be nonempty")
        normalized: dict[str, tuple[str, ...]] = {}
        for name, class_ids in taxonomies.items():
            values = tuple(class_ids)
            if (
                not isinstance(name, str)
                or not name.strip()
                or len(values) < 2
                or any(not isinstance(value, str) or not value.strip() for value in values)
                or len(set(values)) != len(values)
            ):
                raise ValueError("each taxonomy needs a name and unique nonempty class IDs")
            normalized[name] = values
        self.feature_width = feature_width
        self.taxonomy_class_ids = normalized
        self.classifiers = nn.ModuleDict(
            {
                name: nn.Linear(feature_width, len(class_ids))
                for name, class_ids in normalized.items()
            }
        )

    def forward(self, image_features: Tensor, taxonomy: str) -> Tensor:
        if taxonomy not in self.classifiers:
            raise ValueError(f"unknown taxonomy: {taxonomy}")
        if image_features.ndim not in (1, 2) or image_features.shape[-1] != self.feature_width:
            raise ValueError("image features must have shape [width] or [batch, width]")
        if not torch.isfinite(image_features).all():
            raise ValueError("image features must be finite")
        return self.classifiers[taxonomy](image_features)

    @torch.inference_mode()
    def decide(
        self,
        image_features: Tensor,
        taxonomy: str,
        options: list[dict[str, str]],
    ) -> dict[str, object]:
        """Return probabilities normalized over the caller's supplied options."""
        if image_features.ndim != 1:
            raise ValueError("decide expects one image feature vector")
        if taxonomy not in self.taxonomy_class_ids:
            raise ValueError(f"unknown taxonomy: {taxonomy}")
        if not isinstance(options, list) or len(options) < 2:
            raise ValueError("options must contain at least two candidates")
        if any(
            not isinstance(option, dict) or set(option) != {"option_id", "class_id"}
            for option in options
        ):
            raise ValueError("options must contain only option_id and class_id")
        class_ids = self.taxonomy_class_ids[taxonomy]
        supplied = [option["class_id"] for option in options]
        option_ids = [option["option_id"] for option in options]
        if any(not isinstance(value, str) for value in supplied):
            raise ValueError("option class IDs must be strings")
        if len(set(supplied)) != len(supplied) or any(
            class_id not in class_ids for class_id in supplied
        ):
            raise ValueError("options must contain unique known class IDs")
        if any(not isinstance(value, str) or not value.strip() for value in option_ids):
            raise ValueError("options must have nonempty option IDs")
        if len(set(option_ids)) != len(option_ids):
            raise ValueError("options must have unique option IDs")
        logits = self(image_features, taxonomy)
        indexes = torch.tensor(
            [class_ids.index(class_id) for class_id in supplied],
            dtype=torch.long,
            device=logits.device,
        )
        probabilities = torch.softmax(logits[indexes].float(), dim=-1).tolist()
        return classification_json(probabilities, supplied, options)

    @torch.inference_mode()
    def decide_native(self, image_features: Tensor, taxonomy: str, batch) -> dict[str, object]:
        """Score a text-free ObservationBatch and return the shared typed JSON shape."""
        batch, class_ids, abstain_reason = native_classification_request(batch, "image")
        if abstain_reason is not None:
            return native_decision_json(batch, None, abstain_reason=abstain_reason)
        result = self.decide(
            image_features,
            taxonomy,
            [
                {"option_id": option.option_id, "class_id": class_id}
                for option, class_id in zip(batch.options, class_ids, strict=True)
            ],
        )
        return native_decision_json(
            batch,
            [option["probability"] for option in result["options"]],
            selected_option_id=result["selected_option_id"],
        )


def deterministic_asset_split(
    asset_ids: Sequence[str], seed: int, validation_fraction: float
) -> tuple[list[int], list[int]]:
    """Return train/validation indexes using a reproducible hash of asset IDs."""
    if (
        len(asset_ids) < 2
        or any(not isinstance(asset_id, str) or not asset_id.strip() for asset_id in asset_ids)
        or len(set(asset_ids)) != len(asset_ids)
    ):
        raise ValueError("asset IDs must be at least two unique nonempty strings")
    if seed < 0 or not 0 < validation_fraction < 1:
        raise ValueError("seed must be nonnegative and validation fraction must be in (0, 1)")
    validation_count = max(1, min(len(asset_ids) - 1, int(len(asset_ids) * validation_fraction)))
    ranked = sorted(
        range(len(asset_ids)),
        key=lambda index: hashlib.sha256(f"{seed}:{asset_ids[index]}".encode()).digest(),
    )
    validation = set(ranked[:validation_count])
    train_indexes = [index for index in range(len(asset_ids)) if index not in validation]
    validation_indexes = [index for index in range(len(asset_ids)) if index in validation]
    return train_indexes, validation_indexes
