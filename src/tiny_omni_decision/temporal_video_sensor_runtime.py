"""Hash-checked feature-cache adapter for temporal video + sensor decisions."""

from __future__ import annotations

import hashlib
import io
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from .native_feature_decision import TypedOptionSchema
from .observations import ObservationBatch
from .video_sensor_temporal_decision import VideoSensorTemporalDecisionScorer


def _load_series(observation, expected_width: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load one complete immutable NPZ stream and verify its request metadata."""
    if observation.payload_index is not None:
        raise ValueError("temporal feature stream requires a complete cache payload")
    payload = Path(observation.payload_ref).read_bytes()
    if hashlib.sha256(payload).hexdigest() != observation.payload_sha256:
        raise ValueError("temporal feature payload hash mismatch")
    with np.load(io.BytesIO(payload), allow_pickle=False) as cache:
        if set(cache.files) != {"values", "timestamps_ms", "valid_mask"}:
            raise ValueError("temporal feature cache fields mismatch")
        values = cache["values"]
        timestamps = cache["timestamps_ms"]
        valid = cache["valid_mask"]
    if (values.dtype.kind != "f" or values.ndim != 2 or values.shape[1] != expected_width
            or values.shape[0] != len(observation.timestamps_ms)):
        raise ValueError("temporal feature values have incompatible shape or dtype")
    if (timestamps.dtype.kind != "f" or timestamps.shape != (len(values),)
            or not np.array_equal(timestamps, np.asarray(observation.timestamps_ms))):
        raise ValueError("temporal feature timestamps differ from observation metadata")
    if (valid.dtype != np.bool_ or valid.shape != (len(values),)
            or not np.array_equal(valid, np.asarray(observation.valid_mask, dtype=np.bool_))):
        raise ValueError("temporal feature validity differs from observation metadata")
    if not np.isfinite(values[valid]).all():
        raise ValueError("valid temporal feature rows must be finite")
    return values.astype(np.float32, copy=False), timestamps.astype(np.float64, copy=False), valid


def _abstention(option_ids: list[str], missing: list[str]) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "status": "abstain",
        "selected_option_id": None,
        "reason": "insufficient_evidence",
        "missing_modalities": missing,
        "options": [{"option_id": option_id, "probability": None} for option_id in option_ids],
    }


@torch.inference_mode()
def decide_cached_video_sensor_temporal(
    model: VideoSensorTemporalDecisionScorer,
    batch: ObservationBatch,
    *,
    source_by_modality: dict[str, str],
    query_time_ms: float,
    max_age_ms: float,
    required_modalities: Sequence[str],
    sensor_mean: Tensor | Sequence[float],
    sensor_scale: Tensor | Sequence[float],
    option_features: Tensor | None = None,
    option_schema: TypedOptionSchema | None = None,
    context_features: Tensor | None = None,
    class_ids: list[str] | None = None,
    temperature: float = 1.0,
) -> dict[str, object]:
    """Run a temporal Decision from hash-checked video/sensor feature NPZ files.

    Video cache rows are precomputed frozen frame features. Sensor cache rows
    are calibrated physical-channel values; this adapter applies the
    observation's per-channel affine calibration and then train-derived
    ``sensor_mean``/``sensor_scale``. Feature extraction, clock truth,
    calibration provenance, and option/context encoding remain caller-owned.
    Only rows valid at or before ``query_time_ms`` and within ``max_age_ms`` are
    visible to the model. Required stale streams abstain before inference.
    """
    batch = ObservationBatch.model_validate(batch.model_dump())
    if not isinstance(model, VideoSensorTemporalDecisionScorer):
        raise TypeError("model must be a VideoSensorTemporalDecisionScorer")
    if set(source_by_modality) != {"video", "sensor"}:
        raise ValueError("source mapping must contain exactly video and sensor")
    if (any(not isinstance(value, str) or not value.strip()
            for value in source_by_modality.values())
            or len(set(source_by_modality.values())) != 2):
        raise ValueError("video and sensor source IDs must be distinct nonempty strings")
    required = set(required_modalities)
    if not required or not required <= {"video", "sensor"}:
        raise ValueError("required modalities must select video, sensor, or both")
    if (isinstance(query_time_ms, bool) or not isinstance(query_time_ms, (int, float))
            or not math.isfinite(query_time_ms)
            or isinstance(max_age_ms, bool) or not isinstance(max_age_ms, (int, float))
            or not math.isfinite(max_age_ms) or max_age_ms < 0
            or not math.isfinite(temperature) or temperature <= 0):
        raise ValueError("finite query time, nonnegative age and positive temperature required")
    if batch.criterion is not None and batch.criterion.kind != "typed":
        raise ValueError("temporal video-sensor runtime accepts typed criteria only")
    if batch.criterion is not None and context_features is None:
        raise ValueError("typed criterion requires precomputed context features")
    batch.require_shared_clock()

    observations = {observation.source_id: observation for observation in batch.observations}
    if set(observations) != set(source_by_modality.values()):
        raise ValueError("request must contain exactly the configured video and sensor sources")
    by_modality = {
        modality: observations[source_id]
        for modality, source_id in source_by_modality.items()
    }
    for modality, observation in by_modality.items():
        if observation.modality != modality:
            raise ValueError(f"{modality} source has mismatched observation modality")

    if (option_schema is None) == (option_features is None):
        raise ValueError("provide exactly one typed option schema or feature matrix")
    if option_schema is not None:
        option_features = option_schema.encode(batch.options)
    if (not isinstance(option_features, Tensor) or option_features.ndim != 2
            or option_features.shape != (len(batch.options), model.option_width)):
        raise ValueError("typed option features must align with model and supplied options")
    if (context_features is not None
            and (not isinstance(context_features, Tensor)
                 or context_features.shape != (model.context_width,))):
        raise ValueError("context features must match the configured context width")

    option_ids = [option.option_id for option in batch.options]
    classes = list(class_ids) if class_ids is not None else option_ids
    if (len(classes) != len(option_ids)
            or any(not isinstance(value, str) or not value.strip() for value in classes)
            or len(set(classes)) != len(classes)):
        raise ValueError("unique nonempty class IDs must match supplied options")

    video, video_times, video_valid = _load_series(
        by_modality["video"], model.video_feature_width
    )
    sensor, sensor_times, sensor_valid = _load_series(
        by_modality["sensor"], model.sensor_channels
    )
    if len(video) + 1 > 8192 or len(sensor) > 8192:
        raise ValueError("temporal cache exceeds the 8192-row per-stream resource bound")
    channels = by_modality["sensor"].channels
    if len(channels) != model.sensor_channels:
        raise ValueError("sensor calibration channel count differs from model")
    mean = torch.as_tensor(sensor_mean).detach().to(device="cpu", dtype=torch.float64).numpy()
    scale = torch.as_tensor(sensor_scale).detach().to(device="cpu", dtype=torch.float64).numpy()
    if (mean.shape != (model.sensor_channels,) or scale.shape != mean.shape
            or not np.isfinite(mean).all() or not np.isfinite(scale).all()
            or (scale <= 0).any()):
        raise ValueError("sensor normalization must be finite with positive train-derived scales")
    calibration_scale = np.asarray([channel.scale for channel in channels], dtype=np.float64)
    calibration_offset = np.asarray([channel.offset for channel in channels], dtype=np.float64)
    sensor = ((sensor.astype(np.float64) * calibration_scale + calibration_offset - mean)
              / scale).astype(np.float32)
    if not np.isfinite(sensor[sensor_valid]).all():
        raise ValueError("calibrated and normalized sensor values must be finite")

    video_fresh = video_valid & (video_times <= query_time_ms) & (
        query_time_ms - video_times <= max_age_ms
    )
    sensor_fresh = sensor_valid & (sensor_times <= query_time_ms) & (
        query_time_ms - sensor_times <= max_age_ms
    )
    missing = sorted(
        modality for modality, fresh in (("video", video_fresh), ("sensor", sensor_fresh))
        if modality in required and not fresh.any()
    )
    if missing:
        return _abstention(option_ids, missing)

    return model.decide(
        torch.from_numpy(video), torch.from_numpy(video_times),
        torch.from_numpy(video_fresh.astype(np.bool_)),
        torch.from_numpy(sensor), torch.from_numpy(sensor_times),
        torch.from_numpy(sensor_fresh.astype(np.bool_)), option_features,
        option_ids, classes, context_features=context_features, temperature=temperature,
    )
