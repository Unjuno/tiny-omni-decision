"""Bounded HARTH central-label inference; never an instantaneous activity claim."""

import math

import torch
from torch import nn

from tiny_omni_decision.decision_json import (
    decision_json_abstention_record,
    decision_json_record,
)
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


def sensor_window_decision_json(
    result: dict,
    *,
    request_id: str,
    option_ids: list[str],
    option_texts: list[str],
) -> dict[str, object]:
    """Adapt a sensor-window result to the stable caller-ID JSON contract.

    An incomplete window remains an abstention with null probabilities; it is
    never represented as a uniform distribution or a low-confidence answer.
    """
    if not isinstance(result, dict):
        raise ValueError("sensor result must be an object")
    rows = result.get("options")
    if not isinstance(rows, list) or len(rows) != len(option_ids):
        raise ValueError("sensor result options must match caller options")
    if len(option_ids) != len(option_texts):
        raise ValueError("caller option IDs and texts must have matching lengths")
    for row, option_id in zip(rows, option_ids, strict=True):
        if not isinstance(row, dict) or row.get("option_id") != option_id:
            raise ValueError("sensor result option order must match caller option order")

    status = result.get("status")
    if status == "abstain":
        if result.get("selected_option_id") is not None:
            raise ValueError("abstaining sensor result cannot select an option")
        if any(row.get("probability") is not None for row in rows):
            raise ValueError("abstaining sensor result cannot contain probabilities")
        record = decision_json_abstention_record(
            request_id=request_id,
            decision_type="choice",
            option_ids=option_ids,
            option_texts=option_texts,
            reason=result.get("reason"),
        )
    elif status == "selected":
        selected_option_id = result.get("selected_option_id")
        if selected_option_id not in option_ids:
            raise ValueError("selected sensor option is not in caller options")
        record = decision_json_record(
            request_id=request_id,
            decision_type="choice",
            option_ids=option_ids,
            option_texts=option_texts,
            probabilities=[row.get("probability") for row in rows],
            predicted_index=option_ids.index(selected_option_id),
        )
        record["status"] = "selected"
    else:
        raise ValueError("unsupported sensor result status")

    temporal_scope = result.get("temporal_scope")
    if temporal_scope is not None:
        if not isinstance(temporal_scope, dict):
            raise ValueError("temporal_scope must be an object")
        record["temporal_scope"] = temporal_scope
    return record
