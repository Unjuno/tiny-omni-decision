"""Bounded HARTH central-label inference; never an instantaneous activity claim."""

import math

import torch
from torch import nn

from tiny_omni_decision.native_sensor import ACTIVITY_CODES, decide_sensor
from tiny_omni_decision.observations import ObservationBatch


class StaticSensor(nn.Module):
    """Checkpoint-compatible statistics head from the subject-held-out experiment."""

    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(24, 64), nn.GELU(), nn.Linear(64, 12))

    def forward(self, values, timestamps_ms):
        features = torch.cat((values.mean(1), values.std(1), values.amin(1), values.amax(1)), -1)
        return self.head(features)


def decide_sensor_window(model, batch: ObservationBatch, *, query_time_ms, clock_id):
    """Return the center label only after all evidence timestamps have elapsed.

    Evidence end is not measured compute completion or transport availability.
    The caller supplies a query on the observation's clock; no clock conversion.
    """
    batch = ObservationBatch.model_validate(batch.model_dump())
    if len(batch.observations) != 1 or batch.observations[0].modality != "sensor":
        raise ValueError("one sensor observation required")
    observation = batch.observations[0]
    times = observation.timestamps_ms
    if len(times) != 128:
        raise ValueError("exactly 128 samples required")
    if any(
        not math.isclose(b - a, 20.0, abs_tol=0.001) for a, b in zip(times, times[1:], strict=False)
    ):
        raise ValueError("50 Hz sampling required")
    if clock_id != observation.clock_id:
        raise ValueError("query clock must match observation clock")
    if isinstance(query_time_ms, bool) or not math.isfinite(query_time_ms):
        raise ValueError("finite query time required")
    if query_time_ms < times[-1]:
        raise ValueError("complete evidence is not yet available")
    codes = [option.attributes.get("activity_code") for option in batch.options]
    if (
        len(codes) != 12
        or any(type(c) is not int for c in codes)
        or set(codes) != set(ACTIVITY_CODES)
    ):
        raise ValueError("all 12 activity codes required; subsets are unsupported")
    result = decide_sensor(model, batch)
    result["temporal_scope"] = dict(
        task="harth_center_activity",
        clock_id=clock_id,
        query_time_ms=query_time_ms,
        evidence_start_ms=times[0],
        evidence_end_ms=times[-1],
        label_time_ms=times[64],
        future_evidence_ms=times[-1] - times[64],
    )
    return result
