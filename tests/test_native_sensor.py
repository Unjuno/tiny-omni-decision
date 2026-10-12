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
