from __future__ import annotations

import hashlib

import numpy as np
import pytest

torch = pytest.importorskip("torch", exc_type=ImportError)

from tiny_omni_decision.native_feature_decision import (  # noqa: E402
    NumericOptionField,
    TypedOptionSchema,
)
from tiny_omni_decision.observations import ObservationBatch  # noqa: E402
from tiny_omni_decision.sensor_temporal_runtime import (  # noqa: E402
    decide_raw_sensor_observation,
)
from tiny_omni_decision.sensor_temporal_stem import (  # noqa: E402
    SensorTemporalDecisionScorer,
)


def make_batch(tmp_path, *, valid=(True, True, True)):
    path = tmp_path / "sensor.npz"
    values = np.array([[1.0, 2.0], [3.0, 4.0], [1000.0, -1000.0]], dtype=np.float32)
    np.savez(path, values=values)
    payload = path.read_bytes()
    batch = ObservationBatch.model_validate(
        {
            "schema_version": "1.0",
            "observations": [
                {
                    "modality": "sensor",
                    "source_id": "imu",
                    "payload_ref": str(path),
                    "payload_sha256": hashlib.sha256(payload).hexdigest(),
                    "clock_id": "clock-a",
                    "timestamps_ms": (0.0, 10.0, 20.0),
                    "valid_mask": valid,
                    "channels": [
                        {"name": "accel_x", "unit": "g", "scale": 2.0, "offset": 1.0},
                        {"name": "accel_y", "unit": "g", "scale": 0.5, "offset": -1.0},
                    ],
                }
            ],
            "options": [
                {"option_id": "caller-a", "attributes": {"action": 0}},
                {"option_id": "caller-b", "attributes": {"action": 1}},
            ],
        }
    )
    return batch, path, values


def kwargs():
    return {
        "model": SensorTemporalDecisionScorer(
            input_channels=2, option_width=1, hidden_size=8, scorer_hidden_width=16
        ).eval(),
        "option_schema": TypedOptionSchema(numeric=(NumericOptionField("action"),)),
        "expected_channel_names": ("accel_x", "accel_y"),
        "feature_mean": np.array([1.0, 0.0], dtype=np.float32),
        "feature_scale": np.array([2.0, 2.0], dtype=np.float32),
        "query_time_ms": 15.0,
        "max_age_ms": 5.0,
    }


def test_raw_sensor_runtime_hashes_calibrates_and_never_reads_future(tmp_path):
    torch.manual_seed(41)
    batch, _, values = make_batch(tmp_path)
    call = kwargs()

    result = decide_raw_sensor_observation(batch, **call)

    # Only samples at 0 and 10 ms are available at query time 15 ms.
    calibrated = torch.tensor(values[:2] * [2.0, 0.5] + [1.0, -1.0], dtype=torch.float32)
    normalized = (calibrated - torch.tensor([1.0, 0.0])) / torch.tensor([2.0, 2.0])
    expected = call["model"].decide(
        normalized,
        torch.tensor([0.0, 10.0]),
        torch.ones(2, dtype=torch.bool),
        torch.tensor([[0.0], [1.0]]),
        ["caller-a", "caller-b"],
        ["caller-a", "caller-b"],
    )

    assert result == expected
    assert result["status"] == "selected"


def test_raw_sensor_runtime_abstains_for_no_past_or_stale_sensor_values(tmp_path):
    batch, _, _ = make_batch(tmp_path)
    call = kwargs()
    call["query_time_ms"] = -1.0

    no_past = decide_raw_sensor_observation(batch, **call)
    call["query_time_ms"] = 15.0
    call["max_age_ms"] = 4.0
    stale = decide_raw_sensor_observation(batch, **call)

    assert no_past["status"] == stale["status"] == "abstain"
    assert no_past["reason"] == stale["reason"] == "insufficient_evidence"
    assert [row["probability"] for row in stale["options"]] == [None, None]


def test_raw_sensor_runtime_rejects_hash_and_channel_order_mismatches(tmp_path):
    batch, path, _ = make_batch(tmp_path)
    call = kwargs()
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="hash"):
        decide_raw_sensor_observation(batch, **call)

    batch, _, _ = make_batch(tmp_path)
    call["expected_channel_names"] = ("accel_y", "accel_x")
    with pytest.raises(ValueError, match="channel"):
        decide_raw_sensor_observation(batch, **call)


def test_raw_sensor_runtime_ignores_invalid_steps_and_rejects_bad_ranges(tmp_path):
    batch, path, values = make_batch(tmp_path, valid=(True, False, True))
    values[1] = np.nan
    np.savez(path, values=values)
    batch = batch.model_copy(
        update={
            "observations": (
                batch.observations[0].model_copy(
                    update={"payload_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                ),
            )
        }
    )
    call = kwargs()
    call["query_time_ms"] = 20.0
    call["max_age_ms"] = 0.0
    result = decide_raw_sensor_observation(batch, **call)
    assert result["status"] == "selected"

    channels = list(batch.observations[0].channels)
    channels[0] = channels[0].model_copy(update={"valid_range": (-2.0, 2.0)})
    bad_observation = batch.observations[0].model_copy(update={"channels": tuple(channels)})
    bad_batch = batch.model_copy(update={"observations": (bad_observation,)})
    with pytest.raises(ValueError, match="valid range"):
        decide_raw_sensor_observation(bad_batch, **call)
