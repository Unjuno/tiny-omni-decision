"""Verified feature-cache gathering at evidence-availability times."""

import hashlib
import io
from pathlib import Path

import numpy as np

from tiny_omni_decision.evidence_decision import MODALITIES
from tiny_omni_decision.observation_alignment import align_observations
from tiny_omni_decision.observations import ObservationBatch


def gather_aligned_features(batch, source_by_modality, dimensions, query_times_ms, *, max_age_ms):
    """Return [queries,width] features and explicit masks for a fusion caller.

    payload_ref must identify a feature NPZ, not raw media. Its timestamps denote
    when ALL evidence used to encode each feature was available. This code checks
    metadata consistency, not the encoder's causal receptive field or clock truth.
    Absent configured sources are permitted and return false masks. Caller must
    separately abstain when any task-required source is unavailable.
    """
    if (not source_by_modality or set(source_by_modality) != set(dimensions)
            or not set(dimensions) <= MODALITIES
            or any(type(d) is not int or d < 1 for d in dimensions.values())):
        raise ValueError("supported modality slots and positive feature dimensions required")
    sources = list(source_by_modality.values())
    if (len(set(sources)) != len(sources)
            or any(not isinstance(s, str) or not s.strip() for s in sources)):
        raise ValueError("distinct nonempty source IDs required")
    batch = ObservationBatch.model_validate(batch.model_dump())
    times = tuple(query_times_ms)
    aligned = align_observations(batch, times, max_age_ms=max_age_ms)
    observations = {o.source_id: o for o in batch.observations}
    result = {key: {} for key in ("features", "available", "age_ms", "selected_indices")}
    for modality, source_id in source_by_modality.items():
        output = np.zeros((len(times), dimensions[modality]), dtype=np.float32)
        available = np.zeros(len(times), dtype=bool)
        ages = np.full(len(times), np.nan)
        indices = (None,) * len(times)
        if source_id in observations:
            observation = observations[source_id]
            if observation.modality != modality:
                raise ValueError("source modality mismatch")
            if observation.payload_index is not None:
                raise ValueError("feature stream requires an entire cache, not payload_index")
            # Read once so hash validation and decoding refer to identical bytes.
            payload = Path(observation.payload_ref).read_bytes()
            if hashlib.sha256(payload).hexdigest() != observation.payload_sha256:
                raise ValueError("feature payload hash mismatch")
            with np.load(io.BytesIO(payload), allow_pickle=False) as cache:
                if set(cache.files) != {"values", "timestamps_ms", "valid_mask"}:
                    raise ValueError("feature cache fields mismatch")
                values, stamps, valid = cache["values"], cache["timestamps_ms"], cache["valid_mask"]
            if (values.dtype.kind != "f"
                    or values.shape != (len(observation.timestamps_ms), dimensions[modality])):
                raise ValueError("feature cache shape or dtype mismatch")
            if not np.array_equal(stamps, observation.timestamps_ms):
                raise ValueError("feature timestamps differ from observation metadata")
            if valid.dtype != np.bool_ or not np.array_equal(valid, observation.valid_mask):
                raise ValueError("feature validity differs from observation metadata")
            if not np.isfinite(values[valid]).all():
                raise ValueError("nonfinite valid feature")
            stream = aligned[source_id]
            indices = stream["indices"]
            available = np.array(stream["available"], dtype=bool)
            for row, index in enumerate(indices):
                if index is not None:
                    output[row] = values[index]
                    ages[row] = stream["age_ms"][row]
            if not np.isfinite(output).all():
                raise ValueError("feature overflows float32")
        result["features"][modality] = output
        result["available"][modality] = available
        result["age_ms"][modality] = ages
        result["selected_indices"][modality] = indices
    return result
