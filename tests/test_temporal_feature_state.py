from __future__ import annotations

import hashlib

import pytest

from tiny_omni_decision.cache import ObservationFeatureKey
from tiny_omni_decision.temporal_feature_state import (
    SessionTemporalFeatureState,
    TemporalFeatureStateError,
)

RUNTIME = hashlib.sha256(b"runtime-v1").hexdigest()
OTHER_RUNTIME = hashlib.sha256(b"runtime-v2").hexdigest()


def key(modality: str, source_id: str, content: bytes) -> ObservationFeatureKey:
    digest = hashlib.sha256(content).hexdigest()
    return ObservationFeatureKey(
        modality=modality,
        source_id=source_id,
        source_revision="synthetic-revision-1",
        observation_sha256=digest,
        encoder_id=f"synthetic-{modality}",
        encoder_revision="weights-v1",
        encoder_weights_sha256=hashlib.sha256(b"weights-v1").hexdigest(),
        preprocessor_id=f"synthetic-{modality}-preprocessor",
        preprocessor_revision="processor-v1",
        preprocessing_sha256=hashlib.sha256(b"processor-v1").hexdigest(),
        feature_name="synthetic-feature",
        feature_dtype="float32",
        feature_shape=(2,),
        temporal_policy="fixed-two-step-v1" if modality == "video" else None,
    )


def state(**kwargs) -> SessionTemporalFeatureState:
    return SessionTemporalFeatureState(
        runtime_sha256=RUNTIME,
        clock_id="synthetic-clock",
        **kwargs,
    )


def put_feature(store, modality, source_id, timestamp, payload):
    store.update_feature(
        modality=modality,
        source_id=source_id,
        available_at_ms=timestamp,
        key=key(modality, source_id, payload),
        payload=payload,
        runtime_sha256=RUNTIME,
        clock_id="synthetic-clock",
    )


def snapshot(store, at, streams, *, max_age=100.0):
    return store.snapshot(
        as_of_ms=at,
        max_age_ms=max_age,
        required_streams=streams,
        runtime_sha256=RUNTIME,
        clock_id="synthetic-clock",
    )


def test_delayed_queries_reuse_latest_causal_features_per_stream():
    store = state(max_serialized_bytes=16_384, max_events_per_stream=4)
    put_feature(store, "video", "camera", 10.0, b"video-10")
    put_feature(store, "audio", "mic", 12.0, b"audio-12")
    put_feature(store, "video", "camera", 20.0, b"video-20")

    earlier = snapshot(store, 15.0, [("video", "camera"), ("audio", "mic")])
    later = snapshot(store, 22.0, [("video", "camera"), ("audio", "mic")])

    assert earlier["video", "camera"].payload == b"video-10"
    assert earlier["audio", "mic"].payload == b"audio-12"
    assert later["video", "camera"].payload == b"video-20"
    assert later["audio", "mic"].payload == b"audio-12"
    assert store.event_count == 3  # queries do not mutate or re-encode state


def test_explicit_missing_update_masks_older_feature_without_erasing_history():
    store = state(max_serialized_bytes=16_384, max_events_per_stream=4)
    put_feature(store, "video", "camera", 10.0, b"old-video")
    store.record_missing(
        modality="video",
        source_id="camera",
        available_at_ms=20.0,
        runtime_sha256=RUNTIME,
        clock_id="synthetic-clock",
    )

    before = snapshot(store, 15.0, [("video", "camera")])["video", "camera"]
    after = snapshot(store, 21.0, [("video", "camera")])["video", "camera"]

    assert before.payload == b"old-video"
    assert after.payload is None
    assert after.available is False
    assert after.missing_reason == "explicitly_missing"


def test_snapshot_never_reads_future_or_stale_features():
    store = state(max_serialized_bytes=16_384, max_events_per_stream=4)
    put_feature(store, "audio", "mic", 30.0, b"future-audio")

    future = snapshot(store, 29.0, [("audio", "mic")])["audio", "mic"]
    stale = snapshot(store, 50.0, [("audio", "mic")], max_age=10.0)["audio", "mic"]

    assert future.payload is None
    assert future.missing_reason == "not_yet_available"
    assert stale.payload is None
    assert stale.missing_reason == "stale"


@pytest.mark.parametrize("timestamp", [10.0, 9.0])
def test_per_stream_timestamps_must_increase_strictly(timestamp):
    store = state(max_serialized_bytes=16_384, max_events_per_stream=4)
    put_feature(store, "video", "camera", 10.0, b"video-10")

    with pytest.raises(TemporalFeatureStateError, match="strictly increase"):
        put_feature(store, "video", "camera", timestamp, b"video-late")


def test_model_hash_and_clock_mismatch_fail_closed_then_rebind_invalidates():
    store = state(max_serialized_bytes=16_384, max_events_per_stream=4)
    put_feature(store, "video", "camera", 10.0, b"video-10")

    with pytest.raises(TemporalFeatureStateError, match="runtime hash"):
        store.update_feature(
            modality="video",
            source_id="camera",
            available_at_ms=20.0,
            key=key("video", "camera", b"video-20"),
            payload=b"video-20",
            runtime_sha256=OTHER_RUNTIME,
            clock_id="synthetic-clock",
        )
    with pytest.raises(TemporalFeatureStateError, match="clock"):
        store.snapshot(
            as_of_ms=20.0,
            max_age_ms=100.0,
            required_streams=[("video", "camera")],
            runtime_sha256=RUNTIME,
            clock_id="different-clock",
        )
    assert snapshot(store, 20.0, [("video", "camera")])["video", "camera"].payload == b"video-10"

    invalidated = store.rebind(runtime_sha256=OTHER_RUNTIME, clock_id="new-clock")
    assert invalidated == 1
    assert store.bytes_used == 0
    assert store.invalidation_count == 1
    assert store.snapshot(
        as_of_ms=20.0,
        max_age_ms=100.0,
        required_streams=[("video", "camera")],
        runtime_sha256=OTHER_RUNTIME,
        clock_id="new-clock",
    )["video", "camera"].missing_reason == "never_observed"


def test_feature_identity_must_match_stream_and_payload_hash_is_recorded():
    store = state(max_serialized_bytes=16_384, max_events_per_stream=4)
    with pytest.raises(TemporalFeatureStateError, match="identity"):
        store.update_feature(
            modality="video",
            source_id="camera",
            available_at_ms=10.0,
            key=key("video", "other-camera", b"payload"),
            payload=b"payload",
            runtime_sha256=RUNTIME,
            clock_id="synthetic-clock",
        )
    store.update_feature(
        modality="video",
        source_id="camera",
        available_at_ms=10.0,
        key=key("video", "camera", b"raw-observation"),
        payload=b"feature-payload",
        runtime_sha256=RUNTIME,
        clock_id="synthetic-clock",
    )
    entry = snapshot(store, 10.0, [("video", "camera")])["video", "camera"]
    assert entry.payload == b"feature-payload"
    assert entry.payload_sha256 == hashlib.sha256(b"feature-payload").hexdigest()
    assert entry.feature_key.observation_sha256 == hashlib.sha256(b"raw-observation").hexdigest()
    assert store.event_count == 1


def test_event_and_stream_limits_are_enforced_with_visible_fifo_eviction():
    store = state(max_serialized_bytes=16_384, max_events_per_stream=2)
    put_feature(store, "video", "camera", 10.0, b"video-10")
    put_feature(store, "video", "camera", 20.0, b"video-20")
    put_feature(store, "video", "camera", 30.0, b"video-30")

    old = snapshot(store, 15.0, [("video", "camera")])["video", "camera"]
    current = snapshot(store, 35.0, [("video", "camera")])["video", "camera"]
    assert old.payload is None
    assert old.missing_reason == "history_evicted_or_not_observed"
    assert current.payload == b"video-30"
    assert store.event_count == 2
    assert store.eviction_count == 1
    assert store.bytes_used <= store.max_serialized_bytes


def test_global_serialized_byte_budget_evicts_oldest_event_across_streams():
    probe = state(max_serialized_bytes=16_384, max_events_per_stream=4, max_streams=2)
    put_feature(probe, "video", "cam-a", 10.0, b"same-feature")
    one_event_bytes = probe.bytes_used
    store = state(
        max_serialized_bytes=one_event_bytes,
        max_events_per_stream=4,
        max_streams=2,
    )

    put_feature(store, "video", "cam-a", 10.0, b"same-feature")
    put_feature(store, "video", "cam-b", 10.0, b"same-feature")

    first, second = snapshot(
        store, 10.0, [("video", "cam-a"), ("video", "cam-b")]
    ).values()
    assert first.payload is None
    assert first.missing_reason == "history_evicted_or_not_observed"
    assert second.payload == b"same-feature"
    assert store.eviction_count == 1
    assert store.bytes_used == one_event_bytes


def test_payload_over_budget_is_rejected_without_mutating_state():
    store = state(max_serialized_bytes=2_048, max_events_per_stream=4)
    put_feature(store, "audio", "mic", 10.0, b"keep")
    before = snapshot(store, 10.0, [("audio", "mic")])["audio", "mic"].payload
    too_large = b"x" * (store.max_serialized_bytes - store.bytes_used + 1)

    with pytest.raises(TemporalFeatureStateError, match="exceeds state byte budget"):
        put_feature(store, "video", "camera", 20.0, too_large)

    assert snapshot(store, 10.0, [("audio", "mic")])["audio", "mic"].payload == before
    assert store.bytes_used <= store.max_serialized_bytes


def test_maximum_stream_count_is_bounded():
    store = state(max_serialized_bytes=16_384, max_events_per_stream=2, max_streams=1)
    put_feature(store, "audio", "mic", 10.0, b"audio")

    with pytest.raises(TemporalFeatureStateError, match="stream limit"):
        put_feature(store, "video", "camera", 10.0, b"video")
