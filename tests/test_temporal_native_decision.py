# ruff: noqa: E402

from __future__ import annotations

import hashlib

import numpy as np
import pytest

torch = pytest.importorskip("torch", exc_type=ImportError)

from tiny_omni_decision.cache import ObservationFeatureKey  # noqa: E402
from tiny_omni_decision.fusion_decision import FusionDecisionScorer  # noqa: E402
from tiny_omni_decision.native_feature_decision import (
    EnumOptionField,
    TypedOptionSchema,
    decide_temporal_native_fusion,
)  # noqa: E402
from tiny_omni_decision.observations import NativeOption  # noqa: E402
from tiny_omni_decision.temporal_feature_state import SessionTemporalFeatureState  # noqa: E402

RUNTIME = hashlib.sha256(b"runtime-config-v1").hexdigest()
CLOCK = "fixture-common-clock"


def feature_key(
    modality: str,
    source_id: str,
    encoder: str,
    width: int,
    observation: str = "fixture-input",
) -> ObservationFeatureKey:
    return ObservationFeatureKey(
        modality=modality,
        source_id=source_id,
        source_revision="fixture-source-v1",
        observation_sha256=hashlib.sha256(observation.encode()).hexdigest(),
        encoder_id=encoder,
        encoder_revision="fixture-weights-v1",
        encoder_weights_sha256=hashlib.sha256(f"{encoder}-weights".encode()).hexdigest(),
        preprocessor_id=f"{encoder}-processor",
        preprocessor_revision="fixture-processor-v1",
        preprocessing_sha256=hashlib.sha256(f"{encoder}-processor-v1".encode()).hexdigest(),
        feature_name="pooled-feature",
        feature_dtype="float32",
        feature_shape=(width,),
        temporal_policy="fixture-8-frame-v1" if modality == "video" else None,
    )


def put(state, key, values, available_at_ms):
    payload = np.asarray(values, dtype="<f4").tobytes()
    state.update_feature(
        modality=key.modality,
        source_id=key.source_id,
        available_at_ms=available_at_ms,
        key=key,
        payload=payload,
        runtime_sha256=RUNTIME,
        clock_id=CLOCK,
    )


def setup():
    torch.manual_seed(17)
    state = SessionTemporalFeatureState(
        runtime_sha256=RUNTIME,
        clock_id=CLOCK,
        max_serialized_bytes=16_384,
        max_events_per_stream=4,
        max_streams=2,
    )
    keys = {
        "audio": feature_key("audio", "mic-1", "audio-encoder", 2),
        "video": feature_key("video", "camera-1", "video-encoder", 1, "video-10"),
    }
    model = FusionDecisionScorer({"audio": 2, "video": 1}, 2, 1, width=4).eval()
    options = [
        NativeOption(option_id="left", attributes={"direction": "left"}),
        NativeOption(option_id="right", attributes={"direction": "right"}),
    ]
    schema = TypedOptionSchema(
        enums=(EnumOptionField("direction", ("left", "right")),)
    )
    kwargs = {
        "source_by_modality": {"audio": "mic-1", "video": "camera-1"},
        "expected_feature_keys": keys,
        "required_modalities": ("audio", "video"),
        "runtime_sha256": RUNTIME,
        "clock_id": CLOCK,
        "max_age_ms": 100.0,
        "options": options,
        "option_schema": schema,
    }
    return state, keys, model, kwargs


def test_temporal_state_features_reach_native_fusion_at_query_time_without_reencoding():
    state, keys, model, kwargs = setup()
    put(state, keys["audio"], [0.25, -0.5], 12.0)
    put(state, keys["video"], [1.0], 10.0)
    later_video_key = feature_key(
        "video", "camera-1", "video-encoder", 1, "video-20"
    )
    put(state, later_video_key, [9.0], 20.0)

    captured = []
    original_forward = model.forward

    def tracked_forward(features, available, options, option_valid, *args, **kwargs):
        captured.append(
            {
                "features": {name: value.detach().clone() for name, value in features.items()},
                "available": {name: value.detach().clone() for name, value in available.items()},
            }
        )
        return original_forward(features, available, options, option_valid, *args, **kwargs)

    model.forward = tracked_forward
    early = decide_temporal_native_fusion(
        model, state, as_of_ms=15.0, **kwargs
    )
    late = decide_temporal_native_fusion(
        model, state, as_of_ms=22.0, **kwargs
    )

    assert early["status"] == late["status"] == "selected"
    assert [item["option_id"] for item in early["options"]] == ["left", "right"]
    assert sum(item["probability"] for item in late["options"]) == pytest.approx(1.0)
    assert captured[0]["features"]["video"].tolist() == [[1.0]]
    assert captured[1]["features"]["video"].tolist() == [[9.0]]
    assert captured[0]["features"]["audio"].tolist() == [[0.25, -0.5]]
    assert all(item["available"][name].tolist() == [True] for item in captured
               for name in ("audio", "video"))
    assert state.event_count == 3  # inference neither mutates history nor reruns encoders


def test_temporal_fusion_abstains_before_inference_on_stale_or_missing_required_stream():
    state, keys, model, kwargs = setup()
    put(state, keys["audio"], [0.25, -0.5], 10.0)
    kwargs["max_age_ms"] = 5.0
    calls = []
    original_forward = model.forward

    def tracked_forward(*args, **forward_kwargs):
        calls.append(True)
        return original_forward(*args, **forward_kwargs)

    model.forward = tracked_forward
    output = decide_temporal_native_fusion(model, state, as_of_ms=20.0, **kwargs)

    assert output["status"] == "abstain"
    assert output["missing_modalities"] == ["audio", "video"]
    assert all(option["probability"] is None for option in output["options"])
    assert calls == []


def test_temporal_fusion_rejects_feature_key_not_in_frozen_runtime_contract():
    state, keys, model, kwargs = setup()
    put(state, keys["audio"], [0.25, -0.5], 10.0)
    put(state, keys["video"], [1.0], 10.0)
    kwargs["expected_feature_keys"] = {
        **keys,
        "video": feature_key("video", "camera-1", "different-video-encoder", 1),
    }

    with pytest.raises(ValueError, match="expected feature identity"):
        decide_temporal_native_fusion(model, state, as_of_ms=10.0, **kwargs)
