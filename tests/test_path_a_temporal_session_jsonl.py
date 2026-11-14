from __future__ import annotations

import io
import json

import pytest

from tiny_omni_decision.path_a_jsonl import SessionFeatureStateManager, run_jsonl

RUNTIME_SHA256 = "a" * 64
PAYLOAD_BASE64 = "AADAPwAAEMA="
PAYLOAD_SHA256 = "6bfc2c48730924ee3bcd58a6a48a91ef7eef1d7ede12938132f5534418f11cb4"
FEATURE_KEY = {
    "schema_version": 1,
    "modality": "audio",
    "source_id": "mic-1",
    "source_revision": "recording-r1",
    "observation_sha256": "d39ed8b3cea68e3fb9c0e34d3bfdb435c0e09892981141f90a0784b4e694455e",
    "encoder_id": "frozen-audio",
    "encoder_revision": "v1",
    "encoder_weights_sha256": "5fb2ab76ed9bda034b192c48c7069359252fccda168d925acc0ae7d316c0b53e",
    "preprocessor_id": "audio-preprocessor",
    "preprocessor_revision": "p1",
    "preprocessing_sha256": "b10b18f906fcd1ca87847667a5011cd5974171625d4173caf48507d49d5d68ee",
    "feature_name": "pooled",
    "feature_dtype": "float32",
    "feature_shape": [2],
    "temporal_policy": None,
    "start_ms": None,
    "end_ms": None,
}


def observe(at: float, *, clock: str = "clock-1") -> dict:
    return {
        "operation": "observe",
        "session_id": "s1",
        "clock_id": clock,
        "modality": "audio",
        "source_id": "mic-1",
        "available_at_ms": at,
        "feature_key": FEATURE_KEY,
        "feature_payload_base64": PAYLOAD_BASE64,
    }


def snapshot(at: float, max_age: float = 100.0, *, clock: str = "clock-1") -> dict:
    return {
        "operation": "snapshot",
        "session_id": "s1",
        "clock_id": clock,
        "as_of_ms": at,
        "max_age_ms": max_age,
        "required_streams": [{"modality": "audio", "source_id": "mic-1"}],
    }


def execute(records: list[dict], *, manager=None) -> list[dict]:
    if manager is None:
        manager = SessionFeatureStateManager(runtime_sha256=RUNTIME_SHA256)
    source = io.StringIO("".join(json.dumps(record) + "\n" for record in records))
    sink = io.StringIO()
    run_jsonl(source, sink, dispatcher=object(), session_state=manager)
    return [json.loads(line) for line in sink.getvalue().splitlines()]


def test_jsonl_session_snapshot_reuses_latest_causal_feature_payload() -> None:
    outputs = execute([observe(10.0), snapshot(11.0)])

    assert outputs[0]["operation"] == "observe"
    assert outputs[0]["status"] == "stored"
    feature = outputs[1]["features"][0]
    assert feature["available"] is True
    assert feature["age_ms"] == 1.0
    assert feature["payload_sha256"] == PAYLOAD_SHA256
    assert feature["feature_payload_base64"] == PAYLOAD_BASE64


def test_jsonl_snapshot_reports_stale_and_explicit_missing_without_old_payload() -> None:
    missing = {
        "operation": "record_missing",
        "session_id": "s1",
        "clock_id": "clock-1",
        "modality": "audio",
        "source_id": "mic-1",
        "available_at_ms": 20.0,
    }
    outputs = execute([observe(10.0), snapshot(30.0, 5.0), missing, snapshot(21.0)])

    stale, explicit = outputs[1]["features"][0], outputs[3]["features"][0]
    assert (stale["available"], stale["missing_reason"]) == (False, "stale")
    assert "feature_payload_base64" not in stale
    assert (explicit["available"], explicit["missing_reason"]) == (
        False,
        "explicitly_missing",
    )
    assert "feature_payload_base64" not in explicit


def test_jsonl_session_rejects_clock_mismatch_and_unknown_session() -> None:
    with pytest.raises(ValueError, match="clock"):
        execute([observe(10.0), observe(20.0, clock="other-clock")])

    with pytest.raises(ValueError, match="unknown session"):
        execute([snapshot(11.0)])


def test_jsonl_session_limit_requires_explicit_close_before_reuse() -> None:
    manager = SessionFeatureStateManager(runtime_sha256=RUNTIME_SHA256, max_sessions=1)
    first = observe(10.0)
    first["session_id"] = "s1"
    second = observe(10.0)
    second["session_id"] = "s2"
    close = {"operation": "close_session", "session_id": "s1"}

    outputs = execute([first], manager=manager)
    assert outputs[0]["status"] == "stored"
    with pytest.raises(ValueError, match="session limit"):
        execute([second], manager=manager)
    assert execute([close, second], manager=manager)[0]["closed"] is True


def test_invalid_first_observation_does_not_consume_session_capacity() -> None:
    manager = SessionFeatureStateManager(runtime_sha256=RUNTIME_SHA256, max_sessions=1)
    invalid = observe(10.0)
    invalid["feature_payload_base64"] = "not-base64!"

    with pytest.raises(ValueError, match="feature_payload_base64 is invalid"):
        execute([invalid], manager=manager)

    valid = observe(10.0)
    valid["session_id"] = "s2"
    assert execute([valid], manager=manager)[0]["status"] == "stored"


def test_jsonl_session_reports_bounded_history_eviction() -> None:
    records = [observe(float(index)) for index in range(1, 10)]
    records.append(snapshot(1.0))

    outputs = execute(records)

    assert outputs[-2]["evictions"] == 1
    feature = outputs[-1]["features"][0]
    assert feature["available"] is False
    assert feature["missing_reason"] == "history_evicted_or_not_observed"
