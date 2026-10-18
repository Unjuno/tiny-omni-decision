"""Causal metadata alignment with explicit missing/stale masks, no payload reads."""

import math
from bisect import bisect_right

from tiny_omni_decision.observations import ObservationBatch


def align_observations(batch: ObservationBatch, query_times_ms, *, max_age_ms):
    """Hold latest valid observation at/before query within inclusive age limit.

    All streams must already use one verified common clock. No interpolation or
    automatic clock-offset estimation. Absent streams stay absent; unavailable
    samples use None indices, never an index that could gather a future sample.
    Caller must define timestamps as evidence-availability times for causal use;
    e.g. audio-window end, not start, when the whole window informs a feature.
    """
    batch = ObservationBatch.model_validate(batch.model_dump())
    batch.require_shared_clock()
    times = tuple(query_times_ms)
    if not times or any(not math.isfinite(t) for t in times):
        raise ValueError("finite nonempty query times required")
    if any(b <= a for a, b in zip(times, times[1:], strict=False)):
        raise ValueError("query times must strictly increase")
    if not math.isfinite(max_age_ms) or max_age_ms < 0:
        raise ValueError("finite nonnegative maximum age required")
    result = {}
    for observation in batch.observations:
        valid_indices = [i for i, valid in enumerate(observation.valid_mask) if valid]
        valid_times = [observation.timestamps_ms[i] for i in valid_indices]
        indices, ages = [], []
        for query in times:
            position = bisect_right(valid_times, query) - 1
            age = query - valid_times[position] if position >= 0 else None
            available = age is not None and age <= max_age_ms
            indices.append(valid_indices[position] if available else None)
            ages.append(age if available else None)
        result[observation.source_id] = {
            "modality": observation.modality,
            "indices": tuple(indices),
            "age_ms": tuple(ages),
            "available": tuple(i is not None for i in indices),
        }
    return result
