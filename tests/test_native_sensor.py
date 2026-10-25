import hashlib
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from tiny_omni_decision.native_sensor import NativeSensorDecision, decide_sensor  # noqa: E402
from tiny_omni_decision.observations import ObservationBatch  # noqa: E402


def request(tmp_path):
    path = tmp_path / "features.npz"
    np.savez(path, values=np.zeros((1, 128, 6)), timestamps_ms=np.arange(128)[None] * 20.0)
    return ObservationBatch.model_validate(
        dict(
            schema_version="1.0",
            observations=[
                dict(
                    modality="sensor",
                    source_id="s",
                    payload_ref=str(path),
                    payload_index=0,
                    payload_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    clock_id="relative",
                    timestamps_ms=(np.arange(128) * 20.0).tolist(),
                    valid_mask=[True] * 128,
                    channels=[
                        dict(name=n, unit="g", scale=1.0, offset=0.0)
                        for n in ["back_x", "back_y", "back_z", "thigh_x", "thigh_y", "thigh_z"]
                    ],
                )
            ],
            options=[
                dict(option_id="a", attributes={"activity_code": 1}),
                dict(option_id="b", attributes={"activity_code": 6}),
            ],
        )
    )


def test_no_text_json_and_option_identity(tmp_path):
    torch.manual_seed(17)
    model = NativeSensorDecision().eval()
    assert sum(p.numel() for p in model.parameters()) < 1000000
    batch = request(tmp_path)
    result = decide_sensor(model, batch)
    json.dumps(result, allow_nan=False)
    reverse = decide_sensor(model, batch.model_copy(update={"options": batch.options[::-1]}))
    assert result["options"] == reverse["options"][::-1]
    assert result["selected_option_id"] == reverse["selected_option_id"]


def test_missing_signal_abstains(tmp_path):
    batch = request(tmp_path)
    obs = batch.observations[0].model_copy(update={"valid_mask": (False,) * 128})
    result = decide_sensor(
        NativeSensorDecision(), batch.model_copy(update={"observations": (obs,)})
    )
    assert result["status"] == "abstain"
    assert result["selected_option_id"] is None


def test_sensor_decision_uses_typed_native_json_boundary(tmp_path):
    batch = request(tmp_path)
    result = decide_sensor(NativeSensorDecision().eval(), batch)

    assert result["present_modalities"] == ["sensor"]
    assert result["confidence"] == next(
        option["probability"]
        for option in result["options"]
        if option["option_id"] == result["selected_option_id"]
    )
    assert result["options"][0]["attributes"] == {"activity_code": 1}


@pytest.mark.parametrize("change", ["code", "unit", "channel", "criterion"])
def test_unsupported_request_fails_closed(tmp_path, change):
    data = request(tmp_path).model_dump()
    if change == "code":
        data["options"][0]["attributes"]["activity_code"] = 999
    if change == "unit":
        data["observations"][0]["channels"][0]["unit"] = "m/s2"
    if change == "channel":
        data["observations"][0]["channels"][0]["name"] = "unknown"
    if change == "criterion":
        data["criterion"] = {"kind": "text", "value": "different task"}
    with pytest.raises(ValueError):
        decide_sensor(NativeSensorDecision(), ObservationBatch.model_validate(data))


def test_center_runtime_timing_and_full_taxonomy(tmp_path):
    from tiny_omni_decision.native_sensor import ACTIVITY_CODES
    from tiny_omni_decision.native_sensor_runtime import StaticSensor, decide_sensor_window

    data = request(tmp_path).model_dump()
    data["options"] = [
        dict(option_id=f"activity-{c}", attributes={"activity_code": c}) for c in ACTIVITY_CODES
    ]
    batch = ObservationBatch.model_validate(data)
    model = StaticSensor().eval()
    result = decide_sensor_window(model, batch, query_time_ms=2540, clock_id="relative")
    assert result["temporal_scope"]["label_time_ms"] == 1280
    assert result["temporal_scope"]["future_evidence_ms"] == 1260
    json.dumps(result, allow_nan=False)
    reverse = decide_sensor_window(
        model,
        batch.model_copy(update={"options": batch.options[::-1]}),
        query_time_ms=2540,
        clock_id="relative",
    )
    assert result["options"] == reverse["options"][::-1]
    for time, clock in [(1280, "relative"), (2540, "other"), (float("nan"), "relative")]:
        with pytest.raises(ValueError):
            decide_sensor_window(model, batch, query_time_ms=time, clock_id=clock)
    with pytest.raises(ValueError, match="12"):
        decide_sensor_window(model, request(tmp_path), query_time_ms=2540, clock_id="relative")
    obs = batch.observations[0].model_copy(update={"valid_mask": (False,) * 128})
    missing = decide_sensor_window(
        model,
        batch.model_copy(update={"observations": (obs,)}),
        query_time_ms=2540,
        clock_id="relative",
    )
    assert missing["status"] == "abstain"


def test_sensor_window_json_preserves_caller_ids_time_and_abstention(tmp_path):
    from tiny_omni_decision.native_sensor import ACTIVITY_CODES
    from tiny_omni_decision.native_sensor_runtime import (
        StaticSensor,
        decide_sensor_window,
        sensor_window_decision_json,
    )

    data = request(tmp_path).model_dump()
    data["options"] = [
        dict(option_id=f"activity-{code}", attributes={"activity_code": code})
        for code in ACTIVITY_CODES
    ]
    batch = ObservationBatch.model_validate(data)
    texts = [f"activity {code}" for code in ACTIVITY_CODES]
    model = StaticSensor().eval()
    selected = decide_sensor_window(model, batch, query_time_ms=2540, clock_id="relative")
    record = sensor_window_decision_json(
        selected,
        request_id="sensor-17",
        option_ids=[option.option_id for option in batch.options],
        option_texts=texts,
    )
    assert record["status"] == "selected"
    assert record["selected_option_id"] == selected["selected_option_id"]
    assert record["temporal_scope"] == selected["temporal_scope"]
    assert [option["id"] for option in record["options"]] == [
        option.option_id for option in batch.options
    ]
    json.dumps(record, allow_nan=False)

    observation = batch.observations[0].model_copy(update={"valid_mask": (False,) * 128})
    missing = decide_sensor_window(
        model,
        batch.model_copy(update={"observations": (observation,)}),
        query_time_ms=2540,
        clock_id="relative",
    )
    abstention = sensor_window_decision_json(
        missing,
        request_id="sensor-missing",
        option_ids=[option.option_id for option in batch.options],
        option_texts=texts,
    )
    assert abstention["status"] == "abstain"
    assert abstention["selected_option_id"] is None
    assert abstention["temporal_scope"] == missing["temporal_scope"]
    assert all(option["probability"] is None for option in abstention["options"])


def test_sensor_window_json_rejects_option_reordering(tmp_path):
    from tiny_omni_decision.native_sensor import ACTIVITY_CODES
    from tiny_omni_decision.native_sensor_runtime import (
        StaticSensor,
        decide_sensor_window,
        sensor_window_decision_json,
    )

    data = request(tmp_path).model_dump()
    data["options"] = [
        dict(option_id=f"activity-{code}", attributes={"activity_code": code})
        for code in ACTIVITY_CODES
    ]
    batch = ObservationBatch.model_validate(data)
    result = decide_sensor_window(
        StaticSensor().eval(), batch, query_time_ms=2540, clock_id="relative"
    )
    option_ids = [option.option_id for option in batch.options]
    with pytest.raises(ValueError, match="order"):
        sensor_window_decision_json(
            result,
            request_id="sensor-order",
            option_ids=option_ids[::-1],
            option_texts=["activity"] * len(option_ids),
        )
