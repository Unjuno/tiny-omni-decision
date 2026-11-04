"""Text-free Decision adapter for a single cached native observation."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from numbers import Real
from typing import Literal

import numpy as np
import torch

from tiny_omni_decision.decision import FrozenFeatureCandidateScorer
from tiny_omni_decision.decision_json import decision_json_abstention_record
from tiny_omni_decision.observations import Modality, NativeOption, ObservationBatch


def numeric_option_features(
    options: Sequence[NativeOption], field_names: Sequence[str]
) -> np.ndarray:
    """Encode only explicitly selected numeric option attributes, never IDs/text."""
    fields = tuple(field_names)
    if (
        not fields
        or len(set(fields)) != len(fields)
        or any(not isinstance(name, str) or not name.strip() for name in fields)
    ):
        raise ValueError("unique nonempty numeric option fields are required")
    rows = []
    for option in options:
        if set(option.attributes) != set(fields):
            raise ValueError("option attributes must exactly match the frozen numeric schema")
        values = []
        for name in fields:
            value = option.attributes[name]
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError("native option features must be finite numeric attributes")
            values.append(float(value))
        rows.append(values)
    return np.asarray(rows, dtype=np.float32)


def run_text_free_single_modality_decision(
    scorer: FrozenFeatureCandidateScorer,
    batch: ObservationBatch,
    gathered: dict,
    *,
    modality: Modality,
    option_field_names: Sequence[str],
    option_texts: Sequence[str],
    request_id: str,
    decision_type: Literal["choice", "yes_no"] = "choice",
    query_index: int = 0,
    temperature: float = 1.0,
) -> str:
    """Score typed numeric options from one aligned cached feature, without text.

    `gathered` is the validated output of ``gather_aligned_features``. Option
    IDs, display text, source IDs, filenames, and request IDs are used only for
    validation/serialization; none are passed into the scorer.
    """
    batch = ObservationBatch.model_validate(batch.model_dump())
    if batch.criterion is not None or len(batch.observations) != 1:
        raise ValueError("text-free M1 requires exactly one observation and no criterion")
    observation = batch.observations[0]
    if observation.modality != modality or modality == "text":
        raise ValueError("M1 modality must match one non-text observation")
    if isinstance(query_index, bool) or not isinstance(query_index, int) or query_index < 0:
        raise ValueError("query_index must be a nonnegative integer")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")

    option_values = numeric_option_features(batch.options, option_field_names)
    option_ids = [option.option_id for option in batch.options]
    texts = list(option_texts)
    if len(texts) != len(option_ids):
        raise ValueError("option text count must match supplied option count")
    if decision_type == "yes_no" and len(option_ids) != 2:
        raise ValueError("yes_no decisions require exactly two options")

    try:
        feature_values = np.asarray(gathered["features"][modality])
        availability_values = np.asarray(gathered["available"][modality])
    except (KeyError, TypeError) as error:
        raise ValueError("gathered features are missing the requested modality") from error
    if (
        feature_values.dtype.kind != "f"
        or availability_values.dtype != np.bool_
        or feature_values.ndim != 2
        or availability_values.shape != (len(feature_values),)
    ):
        raise ValueError("gathered modality features or availability have invalid shapes")
    features = feature_values.astype(np.float32, copy=False)
    available = availability_values
    if query_index >= len(features):
        raise ValueError("query_index is outside the gathered feature sequence")
    if not available[query_index]:
        return json.dumps(
            decision_json_abstention_record(
                request_id=request_id,
                decision_type=decision_type,
                option_ids=option_ids,
                option_texts=texts,
                reason="insufficient_evidence",
            ),
            ensure_ascii=False,
            allow_nan=False,
        )
    if not np.isfinite(features[query_index]).all():
        raise ValueError("available observation features must be finite")

    expected_size = features.shape[1] + option_values.shape[1]
    if scorer.feature_size != expected_size:
        raise ValueError(f"scorer feature size must be {expected_size}, got {scorer.feature_size}")
    observation_feature = torch.as_tensor(
        features[query_index],
        dtype=torch.float32,
        device=next(scorer.parameters()).device,
    )
    option_tensor = torch.as_tensor(
        option_values, dtype=torch.float32, device=observation_feature.device
    )
    candidate_features = torch.cat(
        (observation_feature.expand(len(option_ids), -1), option_tensor), dim=-1
    )
    with torch.inference_mode():
        output = scorer.decide(candidate_features, temperature=temperature)
    return output.to_decision_json(
        request_id=request_id,
        decision_type=decision_type,
        option_ids=option_ids,
        option_texts=texts,
    )
