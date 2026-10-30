"""Text-free Decision readout over one frozen modality feature and typed options.

This module deliberately accepts numeric feature tensors only. Raw-media encoding,
option-attribute encoding, evidence freshness, and modality-specific abstention stay
in the caller; importing this module never loads a language encoder.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .aligned_features import gather_aligned_features
from .classification_json import classification_json
from .decision import FrozenFeatureCandidateScorer
from .observations import ObservationBatch


@dataclass(frozen=True)
class NumericOptionField:
    """One numeric option attribute with predeclared center and positive scale."""

    name: str
    center: float = 0.0
    scale: float = 1.0

    def __post_init__(self):
        if (
            not self.name.strip()
            or not math.isfinite(self.center)
            or not math.isfinite(self.scale)
            or self.scale <= 0
        ):
            raise ValueError("numeric option fields need a name and finite positive scale")


@dataclass(frozen=True)
class EnumOptionField:
    """One categorical option attribute with a fixed ordered vocabulary."""

    name: str
    values: tuple[str, ...]

    def __post_init__(self):
        if (
            not self.name.strip()
            or len(self.values) < 2
            or any(not isinstance(value, str) or not value.strip() for value in self.values)
            or len(set(self.values)) != len(self.values)
        ):
            raise ValueError("enum option fields need a name and unique fixed categories")


@dataclass(frozen=True)
class TypedOptionSchema:
    """Deterministic encoding for numeric/enumerated attributes, with no text tower."""

    numeric: tuple[NumericOptionField, ...] = ()
    enums: tuple[EnumOptionField, ...] = ()

    def __post_init__(self):
        names = [field.name for field in (*self.numeric, *self.enums)]
        if not names or len(set(names)) != len(names):
            raise ValueError("typed option schema needs unique numeric or enum fields")

    @property
    def feature_width(self) -> int:
        return len(self.numeric) + sum(len(field.values) for field in self.enums)

    def encode(self, options: Sequence) -> Tensor:
        """Encode only declared scalar attributes; IDs and free-form strings are excluded."""
        rows: list[list[float]] = []
        fields = {field.name for field in (*self.numeric, *self.enums)}
        for option in options:
            attributes = option.attributes
            if set(attributes) != fields:
                raise ValueError("option attributes must exactly match the typed schema")
            row: list[float] = []
            for field in self.numeric:
                value = attributes[field.name]
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError("numeric option attributes must be numbers, not booleans/text")
                normalized = (float(value) - field.center) / field.scale
                if not math.isfinite(normalized):
                    raise ValueError("numeric option attribute must be finite")
                row.append(normalized)
            for field in self.enums:
                value = attributes[field.name]
                if not isinstance(value, str) or value not in field.values:
                    raise ValueError("enum option attribute is outside its fixed vocabulary")
                row.extend(float(value == category) for category in field.values)
            rows.append(row)
        if len(rows) < 2:
            raise ValueError("at least two typed options are required")
        return torch.tensor(rows, dtype=torch.float32)


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
        if observation_features.ndim != 2 or observation_features.shape[-1] != self.feature_width:
            raise ValueError("observation features must have shape [batch, feature_width]")
        if (
            option_features.ndim != 3
            or option_features.shape[0] != observation_features.shape[0]
            or option_features.shape[-1] != self.option_width
        ):
            raise ValueError("option features must have shape [batch, candidates, option_width]")
        if (
            option_valid.dtype != torch.bool
            or option_valid.shape != option_features.shape[:2]
            or not (option_valid.sum(-1) >= 2).all()
        ):
            raise ValueError("a boolean mask with at least two options per row is required")
        if not observation_features.is_floating_point() or not option_features.is_floating_point():
            raise ValueError("observation and option features must use floating-point tensors")
        if (
            observation_features.device != option_features.device
            or option_valid.device != option_features.device
            or observation_features.dtype != option_features.dtype
        ):
            raise ValueError("features and validity mask must share device and feature dtype")
        if (
            not torch.isfinite(observation_features).all()
            or not torch.isfinite(option_features[option_valid]).all()
        ):
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
        if (
            len(option_ids) != len(option_features)
            or len(class_ids) != len(option_features)
            or any(
                not isinstance(value, str) or not value.strip()
                for value in [*option_ids, *class_ids]
            )
            or len(set(option_ids)) != len(option_ids)
            or len(set(class_ids)) != len(class_ids)
        ):
            raise ValueError("unique nonempty option and class IDs must match the option rows")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")

        reference = next(self.parameters())
        observation_features = observation_features.to(
            device=reference.device, dtype=reference.dtype
        )
        option_features = option_features.to(device=reference.device, dtype=reference.dtype)
        logits = (
            self(
                observation_features[None],
                option_features[None],
                torch.ones(
                    (1, len(option_features)), dtype=torch.bool, device=option_features.device
                ),
            )[0].float()
            / temperature
        )
        probabilities = torch.softmax(logits, dim=-1).cpu().tolist()
        options = [
            {"option_id": option_id, "class_id": class_id}
            for option_id, class_id in zip(option_ids, class_ids, strict=True)
        ]
        return classification_json(probabilities, class_ids, options)


@torch.inference_mode()
def decide_cached_native_feature(
    model: NativeFeatureDecisionScorer,
    batch: ObservationBatch,
    option_features: Tensor | None = None,
    class_ids: list[str] | None = None,
    *,
    modality: str,
    query_time_ms: float,
    max_age_ms: float,
    option_schema: TypedOptionSchema | None = None,
    feature_mean: Tensor | None = None,
    feature_scale: Tensor | None = None,
    temperature: float = 1.0,
) -> dict[str, object]:
    """Run a text-free M1 decision from one hash-checked cached feature stream.

    Candidates are encoded from ``batch.options`` using exactly one of a fixed
    typed schema or a precomputed task-owned feature matrix. This adapter rejects
    criteria and mixed-modality
    requests so a text or second stream cannot silently enter an M1 result.
    Missing or stale evidence produces an abstention with null probabilities.
    """
    batch = ObservationBatch.model_validate(batch.model_dump())
    if modality not in {"audio", "image", "screen", "video", "sensor"}:
        raise ValueError("M1 requires one supported non-text modality")
    if batch.criterion is not None:
        raise ValueError("single-modality M1 adapter does not accept a criterion")
    if len(batch.observations) != 1 or batch.observations[0].modality != modality:
        raise ValueError("exactly one observation of the selected modality is required")
    if (option_schema is None) == (option_features is None):
        raise ValueError("provide exactly one typed option schema or option feature matrix")
    if option_schema is not None:
        option_features = option_schema.encode(batch.options)
    if (
        option_features is None
        or option_features.ndim != 2
        or option_features.shape[0] != len(batch.options)
    ):
        raise ValueError("typed option features must align with supplied options")
    if (feature_mean is None) != (feature_scale is None):
        raise ValueError("feature normalization requires both mean and scale")
    reference = next(model.parameters())
    if feature_mean is not None and feature_scale is not None:
        feature_mean = torch.as_tensor(feature_mean, device=reference.device, dtype=reference.dtype)
        feature_scale = torch.as_tensor(
            feature_scale, device=reference.device, dtype=reference.dtype
        )
        if (
            feature_mean.shape != (model.feature_width,)
            or feature_scale.shape != (model.feature_width,)
            or not torch.isfinite(feature_mean).all()
            or not torch.isfinite(feature_scale).all()
            or (feature_scale <= 0).any()
        ):
            raise ValueError(
                "feature normalization must be finite width-sized mean and positive scale"
            )
    option_ids = [option.option_id for option in batch.options]
    classes = list(class_ids) if class_ids is not None else option_ids
    # Validate class coverage and option identifiers before reading feature data.
    if len(classes) != len(option_ids):
        raise ValueError("one class ID is required per supplied option")
    classification_json(
        [1.0 / len(classes)] * len(classes) if classes else [],
        classes,
        [
            {"option_id": option_id, "class_id": class_id}
            for option_id, class_id in zip(option_ids, classes, strict=True)
        ],
    )

    observation = batch.observations[0]
    gathered = gather_aligned_features(
        batch,
        {modality: observation.source_id},
        {modality: model.feature_width},
        [query_time_ms],
        max_age_ms=max_age_ms,
    )
    if not gathered["available"][modality][0]:
        return {
            "schema_version": "1.0",
            "status": "abstain",
            "selected_option_id": None,
            "reason": "insufficient_evidence",
            "missing_modalities": [modality],
            "options": [{"option_id": option_id, "probability": None} for option_id in option_ids],
        }

    features = torch.from_numpy(gathered["features"][modality][0]).to(
        device=reference.device, dtype=reference.dtype
    )
    if feature_mean is not None and feature_scale is not None:
        features = (features - feature_mean) / feature_scale
    return model.decide(
        features,
        option_features,
        option_ids,
        classes,
        temperature=temperature,
    )
