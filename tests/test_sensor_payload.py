import hashlib

import numpy as np
import pytest

from tiny_omni_decision.observations import Observation
from tiny_omni_decision.sensor_payload import load_sensor_payload


def fixture(tmp_path, extra=None):
    path = tmp_path / "features.npz"
    np.savez(
        path,
        values=np.array([[[1.0], [2.0]]]),
        timestamps_ms=np.array([[0.0, 20.0]]),
        **(extra or {}),
    )
    return Observation(
        modality="sensor",
        source_id="s",
        payload_ref=str(path),
        payload_index=0,
        payload_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        clock_id="relative",
        timestamps_ms=[0.0, 20.0],
        valid_mask=[True, False],
        channels=[dict(name="a", unit="g", scale=2.0, offset=1.0)],
    )


def test_verified_payload_calibration_and_mask(tmp_path):
    loaded = load_sensor_payload(fixture(tmp_path))
    assert loaded.values.tolist() == [[3.0], [0.0]]
    assert loaded.valid_mask.tolist() == [True, False]
    assert not loaded.values.flags.writeable


@pytest.mark.parametrize(
    "case", ["hash", "time", "index", "missing_index", "channels", "target", "budget"]
)
def test_payload_rejects_mismatch(tmp_path, case):
    observation = fixture(tmp_path, {"label": np.array([1])} if case == "target" else None)
    updates = {
        "hash": {"payload_sha256": "0" * 64},
        "time": {"timestamps_ms": (0.0, 21.0)},
        "index": {"payload_index": 1},
        "missing_index": {"payload_index": None},
        "channels": {"channels": observation.channels * 2},
    }
    observation = observation.model_copy(update=updates.get(case, {}))
    with pytest.raises(ValueError):
        load_sensor_payload(observation, max_bytes=1 if case == "budget" else 1024**2)
