import json

import pytest
from pydantic import ValidationError

from tiny_omni_decision.observations import ObservationBatch
from tiny_omni_decision.sensor_windows import contiguous_windows


def batch(modality="sensor"):
    observation = dict(
        modality=modality,
        source_id="stream-1",
        payload_ref="local:fixture",
        payload_sha256="a" * 64,
        timestamps_ms=[0.0, 20.0, 40.0],
        valid_mask=[True] * 3,
        clock_id="session-relative-ms",
    )
    if modality == "sensor":
        observation["channels"] = [dict(name="acc_x", unit="g", scale=1.0, offset=0.0)]
    return dict(
        schema_version="1.0",
        observations=[observation],
        options=[
            dict(option_id="left", attributes={"action_code": 0}),
            dict(option_id="right", attributes={"action_code": 1}),
        ],
    )


@pytest.mark.parametrize("modality", ["audio", "video", "image", "screen", "sensor"])
def test_native_inputs_need_no_text(modality):
    value = ObservationBatch.model_validate(batch(modality))
    assert value.criterion is None
    assert value.present_modalities == (modality,)
    assert ObservationBatch.model_validate_json(value.model_dump_json()) == value
    assert "torch" not in json.dumps(value.model_dump())


@pytest.mark.parametrize("case", ["duplicate_option", "mask", "nan", "backwards", "unit", "empty"])
def test_rejects_invalid_observations(case):
    data = batch()
    obs = data["observations"][0]
    if case == "duplicate_option":
        data["options"][1]["option_id"] = "left"
    if case == "mask":
        obs["valid_mask"] = [True]
    if case == "nan":
        obs["timestamps_ms"][0] = float("nan")
    if case == "backwards":
        obs["timestamps_ms"] = [20.0, 0.0, 40.0]
    if case == "unit":
        obs["channels"][0]["unit"] = ""
    if case == "empty":
        data["observations"] = []
    with pytest.raises(ValidationError):
        ObservationBatch.model_validate(data)


def test_invalid_present_stream_is_not_absent():
    data = batch()
    data["observations"][0]["valid_mask"] = [False] * 3
    value = ObservationBatch.model_validate(data)
    assert value.present_modalities == ("sensor",)
    assert not any(value.observations[0].valid_mask)


def test_windows_do_not_cross_gaps_or_invalid_samples():
    assert contiguous_windows(
        [0.0, 20.0, 40.0, 1500.0, 1520.0, 1540.0], [True] * 6, length=3, stride=1, max_gap_ms=21.0
    ) == [(0, 3), (3, 6)]
    assert contiguous_windows(
        [0.0, 20.0, 40.0, 60.0, 80.0],
        [True, True, False, True, True],
        length=2,
        stride=1,
        max_gap_ms=21.0,
    ) == [(0, 2), (3, 5)]


@pytest.mark.parametrize("times", [[0.0, 0.0], [20.0, 0.0], [0.0, float("nan")]])
def test_windows_fail_closed_on_invalid_time(times):
    with pytest.raises(ValueError):
        contiguous_windows(times, [True] * 2, length=2, stride=1, max_gap_ms=21.0)


def test_synchronized_fusion_and_spoken_reference_validation():
    data = batch("video")
    audio = batch("audio")["observations"][0]
    audio["source_id"] = "audio-1"
    data["observations"].append(audio)
    data["criterion"] = {"kind": "spoken", "value": "audio-1"}
    value = ObservationBatch.model_validate(data)
    value.require_shared_clock()
    assert value.present_modalities == ("audio", "video")
    data["observations"][1]["clock_id"] = "other-clock"
    with pytest.raises(ValueError, match="alignment"):
        ObservationBatch.model_validate(data).require_shared_clock()
    data["criterion"]["value"] = "unknown"
    with pytest.raises(ValidationError):
        ObservationBatch.model_validate(data)


def test_observation_contract_imports_no_encoder():
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import tiny_omni_decision.observations; "
            'assert "torch" not in sys.modules; assert "transformers" not in sys.modules',
        ],
        check=True,
    )


def test_payload_index_is_explicit_and_nonnegative():
    data = batch()
    data["observations"][0]["payload_index"] = 0
    assert ObservationBatch.model_validate(data).observations[0].payload_index == 0
    data["observations"][0]["payload_index"] = -1
    with pytest.raises(ValidationError):
        ObservationBatch.model_validate(data)
