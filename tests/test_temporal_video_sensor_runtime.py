from __future__ import annotations

import hashlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def request(tmp_path, *, clock_sensor="shared", sensor_valid=(True, True, False, True)):
    from tiny_omni_decision.observations import ObservationBatch

    video_path = tmp_path / "video.npz"
    video_values = np.arange(18, dtype=np.float32).reshape(3, 6) / 10
    video_times = np.array([100.0, 200.0, 300.0], dtype=np.float64)
    video_mask = np.array([True, True, True], dtype=np.bool_)
    np.savez(video_path, values=video_values, timestamps_ms=video_times,
             valid_mask=video_mask)

    sensor_path = tmp_path / "sensor.npz"
    sensor_values = np.arange(12, dtype=np.float32).reshape(4, 3)
    sensor_times = np.array([90.0, 190.0, 290.0, 390.0], dtype=np.float64)
    sensor_mask = np.asarray(sensor_valid, dtype=np.bool_)
    np.savez(sensor_path, values=sensor_values, timestamps_ms=sensor_times,
             valid_mask=sensor_mask)

    batch = ObservationBatch.model_validate({
        "schema_version": "1.0",
        "observations": [
            {"modality": "video", "source_id": "cam", "payload_ref": str(video_path),
             "payload_sha256": hashlib.sha256(video_path.read_bytes()).hexdigest(),
             "clock_id": "shared", "timestamps_ms": tuple(map(float, video_times)),
             "valid_mask": tuple(map(bool, video_mask))},
            {"modality": "sensor", "source_id": "imu", "payload_ref": str(sensor_path),
             "payload_sha256": hashlib.sha256(sensor_path.read_bytes()).hexdigest(),
             "clock_id": clock_sensor, "timestamps_ms": tuple(map(float, sensor_times)),
             "valid_mask": tuple(map(bool, sensor_mask)), "channels": [
                 {"name": "x", "unit": "m/s2", "scale": 2.0, "offset": 1.0},
                 {"name": "y", "unit": "m/s2", "scale": 1.0, "offset": -1.0},
                 {"name": "z", "unit": "m/s2", "scale": 0.5, "offset": 0.0},
             ]},
        ],
        "options": [
            {"option_id": "choice-2", "attributes": {"action": "walk"}},
            {"option_id": "choice-1", "attributes": {"action": "run"}},
        ],
        "criterion": {"kind": "typed", "value": "activity"},
    })
    return batch, video_path, sensor_path, video_values, sensor_values


def make_model():
    from tiny_omni_decision.video_sensor_temporal_decision import (
        VideoSensorTemporalDecisionScorer,
    )

    torch.manual_seed(177)
    return VideoSensorTemporalDecisionScorer(
        6, 3, 2, context_width=2, sensor_hidden_size=5,
        width=8, heads=2, scorer_hidden_width=7,
    ).eval()


def invoke(model, batch, *, max_age_ms=250.0, required=("video", "sensor")):
    from tiny_omni_decision.native_feature_decision import (
        EnumOptionField,
        TypedOptionSchema,
    )
    from tiny_omni_decision.temporal_video_sensor_runtime import (
        decide_cached_video_sensor_temporal,
    )

    return decide_cached_video_sensor_temporal(
        model,
        batch,
        source_by_modality={"video": "cam", "sensor": "imu"},
        query_time_ms=300.0,
        max_age_ms=max_age_ms,
        required_modalities=required,
        option_schema=TypedOptionSchema(
            enums=(EnumOptionField("action", ("walk", "run")),)
        ),
        context_features=torch.tensor([0.25, -0.5]),
        sensor_mean=torch.zeros(3),
        sensor_scale=torch.ones(3),
    )


def test_verified_temporal_cache_reaches_lstm_and_returns_supplied_option_json(tmp_path):
    batch, _, _, _, _ = request(tmp_path)
    model = make_model()
    seen = {}
    original = model.decide

    def track_decide(*args, **kwargs):
        seen["video"] = args[0].clone()
        seen["video_valid"] = args[2].clone()
        seen["sensor"] = args[3].clone()
        seen["sensor_valid"] = args[5].clone()
        seen["context"] = kwargs["context_features"].clone()
        return original(*args, **kwargs)

    model.decide = track_decide
    result = invoke(model, batch)
    assert result["status"] == "selected"
    assert [item["option_id"] for item in result["options"]] == ["choice-2", "choice-1"]
    assert sum(item["probability"] for item in result["options"]) == pytest.approx(1.0)
    assert seen["video"].shape == (3, 6)
    assert seen["video_valid"].tolist() == [True, True, True]
    assert seen["sensor"].tolist() == [
        [1.0, 0.0, 1.0], [7.0, 3.0, 2.5], [13.0, 6.0, 4.0], [19.0, 9.0, 5.5]
    ]
    assert seen["sensor_valid"].tolist() == [True, True, False, False]
    torch.testing.assert_close(seen["context"], torch.tensor([0.25, -0.5]))


def test_stale_required_stream_abstains_before_temporal_inference(tmp_path):
    batch, *_ = request(tmp_path)
    model = make_model()
    model.decide = lambda *args, **kwargs: pytest.fail("stale evidence must abstain")
    result = invoke(model, batch, max_age_ms=50.0)
    assert result["status"] == "abstain"
    assert result["missing_modalities"] == ["sensor"]
    assert all(item["probability"] is None for item in result["options"])


def test_stale_optional_sensor_is_masked_while_fresh_video_can_decide(tmp_path):
    batch, *_ = request(tmp_path)
    model = make_model()
    seen = {}
    original = model.decide

    def track_decide(*args, **kwargs):
        seen["sensor_valid"] = args[5].clone()
        return original(*args, **kwargs)

    model.decide = track_decide
    result = invoke(model, batch, max_age_ms=50.0, required=("video",))
    assert result["status"] == "selected"
    assert seen["sensor_valid"].tolist() == [False, False, False, False]


@pytest.mark.parametrize("case", ["payload_hash", "clock_mismatch", "future_context"])
def test_temporal_runtime_rejects_untrusted_or_incompatible_inputs(tmp_path, case):
    batch, _, sensor_path, _, _ = request(
        tmp_path, clock_sensor="different" if case == "clock_mismatch" else "shared"
    )
    if case == "payload_hash":
        sensor_path.write_bytes(sensor_path.read_bytes() + b"tampered")
    model = make_model()
    if case == "future_context":
        from tiny_omni_decision.temporal_video_sensor_runtime import (
            decide_cached_video_sensor_temporal,
        )
        result = decide_cached_video_sensor_temporal(
            model, batch, source_by_modality={"video": "cam", "sensor": "imu"},
            query_time_ms=50.0, max_age_ms=500.0, required_modalities=("video",),
            option_features=torch.eye(2), context_features=torch.zeros(2),
            sensor_mean=torch.zeros(3), sensor_scale=torch.ones(3),
        )
        assert result["status"] == "abstain"
        assert result["missing_modalities"] == ["video"]
    else:
        with pytest.raises(ValueError):
            invoke(model, batch)
