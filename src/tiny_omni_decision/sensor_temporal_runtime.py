"""Integrity-checked raw sensor observations for the compact temporal Decision path."""

import hashlib
import io
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from .native_feature_decision import TypedOptionSchema
from .observations import ObservationBatch
from .sensor_temporal_stem import SensorTemporalDecisionScorer


def _abstain(option_ids: Sequence[str]) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "status": "abstain",
        "selected_option_id": None,
        "reason": "insufficient_evidence",
        "missing_modalities": ["sensor"],
        "options": [{"option_id": value, "probability": None} for value in option_ids],
    }


@torch.inference_mode()
def decide_raw_sensor_observation(
    batch: ObservationBatch,
    *,
    model: SensorTemporalDecisionScorer,
    option_schema: TypedOptionSchema,
    expected_channel_names: Sequence[str],
    feature_mean: Sequence[float],
    feature_scale: Sequence[float],
    query_time_ms: float,
    max_age_ms: float,
    temperature: float = 1.0,
) -> dict[str, object]:
    """Run text-free Decision on one hash-checked raw sensor NPZ.

    The NPZ must contain exactly ``values`` with shape ``[time, channels]``.
    Timestamps, validity, units and affine sensor calibration come from the
    validated observation contract. Only valid readings at or before the query
    time are passed to the causal LSTM. Normalization statistics are supplied
    by the caller and must have been fit on training data only.
    """
    batch = ObservationBatch.model_validate(batch.model_dump())
    if not isinstance(model, SensorTemporalDecisionScorer):
        raise TypeError("model must be a SensorTemporalDecisionScorer")
    if batch.criterion is not None or len(batch.observations) != 1:
        raise ValueError("raw sensor M1 accepts one sensor observation and no criterion")
    observation = batch.observations[0]
    if observation.modality != "sensor":
        raise ValueError("raw sensor M1 requires a sensor observation")
    if observation.payload_index is not None:
        raise ValueError("raw sensor M1 requires a standalone NPZ payload")
    if not math.isfinite(query_time_ms) or not math.isfinite(max_age_ms) or max_age_ms < 0:
        raise ValueError("query time must be finite and max_age_ms finite and nonnegative")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")

    names = tuple(expected_channel_names)
    channel_names = tuple(channel.name for channel in observation.channels)
    if (len(names) != model.stem.input_channels or len(set(names)) != len(names)
            or names != channel_names):
        raise ValueError("sensor channel names/order do not match the configured model")
    if (len(batch.options) < 2 or option_schema.feature_width != model.readout.option_width):
        raise ValueError("typed option schema must match the model and provide options")
    option_ids = [option.option_id for option in batch.options]
    option_features = option_schema.encode(batch.options)

    payload = Path(observation.payload_ref).read_bytes()
    if hashlib.sha256(payload).hexdigest() != observation.payload_sha256:
        raise ValueError("raw sensor payload hash mismatch")
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        if set(archive.files) != {"values"}:
            raise ValueError("raw sensor NPZ must contain exactly the values array")
        raw = archive["values"]
    if (raw.shape != (len(observation.timestamps_ms), len(names))
            or raw.dtype.kind not in "fiu"):
        raise ValueError("raw sensor values shape or numeric dtype mismatch")

    timestamps = np.asarray(observation.timestamps_ms, dtype=np.float64)
    valid = np.asarray(observation.valid_mask, dtype=bool) & (timestamps <= query_time_ms)
    selected_indices = np.flatnonzero(valid)
    if selected_indices.size == 0:
        return _abstain(option_ids)
    age_ms = query_time_ms - timestamps[selected_indices[-1]]
    if age_ms > max_age_ms:
        return _abstain(option_ids)

    means = np.asarray(feature_mean, dtype=np.float64)
    scales = np.asarray(feature_scale, dtype=np.float64)
    if (means.shape != (len(names),) or scales.shape != means.shape
            or not np.isfinite(means).all() or not np.isfinite(scales).all()
            or (scales <= 0).any()):
        raise ValueError("training feature normalization must be finite and positive")

    selected = raw[selected_indices].astype(np.float64, copy=False)
    if not np.isfinite(selected).all():
        raise ValueError("valid raw sensor values must be finite")
    calibration = np.asarray([channel.scale for channel in observation.channels])
    offsets = np.asarray([channel.offset for channel in observation.channels])
    calibrated = selected * calibration + offsets
    if not np.isfinite(calibrated).all():
        raise ValueError("calibrated sensor values must be finite")
    for index, channel in enumerate(observation.channels):
        if channel.valid_range is not None:
            low, high = channel.valid_range
            if ((calibrated[:, index] < low) | (calibrated[:, index] > high)).any():
                raise ValueError("calibrated sensor value is outside its declared valid range")
    normalized = ((calibrated - means) / scales).astype(np.float32)
    if not np.isfinite(normalized).all():
        raise ValueError("normalized sensor values must fit finite float32")

    reference = next(model.parameters())
    values = torch.as_tensor(normalized, device=reference.device, dtype=reference.dtype)
    times = torch.as_tensor(
        timestamps[selected_indices], device=reference.device, dtype=reference.dtype
    )
    mask = torch.ones(len(selected_indices), device=reference.device, dtype=torch.bool)
    return model.decide(
        values,
        times,
        mask,
        option_features,
        option_ids,
        option_ids,
        temperature=temperature,
    )
