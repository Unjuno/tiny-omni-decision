"""Text-free single-stream decisions over hash-verified frozen feature caches."""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Real

import torch

from tiny_omni_decision.aligned_features import gather_aligned_features
from tiny_omni_decision.decision import FrozenFeatureCandidateScorer
from tiny_omni_decision.decision_json import (
    decision_json_abstention_record,
    decision_json_record,
)
from tiny_omni_decision.observations import ObservationBatch


def decide_single_modality_from_cache(
    scorer: FrozenFeatureCandidateScorer,
    batch: ObservationBatch,
    *,
    request_id: str,
    feature_width: int,
    option_attribute_keys: Sequence[str],
    temperature: float = 1.0,
) -> dict[str, object]:
    """Pool one hash-verified feature stream and score numeric typed options.

    This is a text-free masked-mean Path A baseline. The supplied options must
    carry the same ordered numeric attribute schema; text and implicit query
    encoders are never loaded. Temporal modeling belongs to a separate scorer.
    """
    batch = ObservationBatch.model_validate(batch.model_dump())
    if len(batch.observations) != 1:
        raise ValueError("exactly one modality observation is required")
    if batch.criterion is not None:
        raise ValueError("text-free single-modality scoring does not accept a criterion")
    if type(feature_width) is not int or feature_width < 1:
        raise ValueError("positive feature width required")
    keys = tuple(option_attribute_keys)
    if not keys or any(not isinstance(key, str) or not key.strip() for key in keys):
        raise ValueError("nonempty typed option attribute keys required")
    if len(set(keys)) != len(keys):
        raise ValueError("typed option attribute keys must be unique")
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, Real)
        or not math.isfinite(temperature)
        or temperature <= 0
    ):
        raise ValueError("temperature must be finite and positive")

    attribute_rows = []
    for option in batch.options:
        if set(option.attributes) != set(keys):
            raise ValueError("every option must provide exactly the typed numeric attributes")
        row = []
        for key in keys:
            value = option.attributes[key]
            if isinstance(value, str) or not isinstance(value, (bool, int, float)):
                raise ValueError("option attributes must be numeric, never text")
            number = float(value)
            if not math.isfinite(number):
                raise ValueError("option attributes must be finite numeric values")
            row.append(number)
        attribute_rows.append(row)

    observation = batch.observations[0]
    gathered = gather_aligned_features(
        batch,
        {observation.modality: observation.source_id},
        {observation.modality: feature_width},
        observation.timestamps_ms,
        max_age_ms=0.0,
    )
    available = gathered["available"][observation.modality]
    option_ids = [option.option_id for option in batch.options]
    if not available.any():
        return decision_json_abstention_record(
            request_id=request_id,
            decision_type="choice",
            option_ids=option_ids,
            option_texts=option_ids,
            reason="missing_or_invalid_observation",
        ) | {"status": "abstain", "modality": observation.modality}

    if scorer.training:
        raise ValueError("single-modality inference requires an eval-mode scorer")
    if scorer.feature_size != feature_width + len(keys):
        raise ValueError("scorer width must match observation plus typed option features")
    try:
        parameter = next(scorer.parameters())
    except StopIteration as exc:
        raise ValueError("scorer must have model parameters") from exc

    feature_rows = torch.as_tensor(
        gathered["features"][observation.modality],
        dtype=parameter.dtype,
        device=parameter.device,
    )
    valid = torch.as_tensor(available, dtype=torch.bool, device=parameter.device)
    pooled = feature_rows[valid].mean(dim=0)
    option_features = torch.tensor(attribute_rows, dtype=parameter.dtype, device=parameter.device)
    paired = torch.cat((pooled.expand(len(batch.options), -1), option_features), dim=-1)
    with torch.inference_mode():
        output = scorer.decide(paired, temperature=temperature)

    result = decision_json_record(
        request_id=request_id,
        decision_type="choice",
        option_ids=option_ids,
        option_texts=option_ids,
        probabilities=output.probabilities.detach().cpu().tolist(),
        predicted_index=output.predicted_index,
        confidence=output.confidence,
    )
    return result | {"status": "selected", "modality": observation.modality}
