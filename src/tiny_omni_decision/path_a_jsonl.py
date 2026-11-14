"""JSONL entry point for the frozen-feature Path A decision runtime."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import struct
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any, TextIO

from .cache import FeatureCacheError, ObservationFeatureKey
from .observations import OBSERVATION_MODALITIES
from .runtime_dispatch import TypedDecisionSpec
from .temporal_feature_state import (
    SessionTemporalFeatureState,
    TemporalFeatureStateError,
)


def _string_tuple(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{name} must be a JSON array of strings")
    return tuple(value)


def _decision_spec(value: Any) -> TypedDecisionSpec:
    if not isinstance(value, dict):
        raise ValueError("each decision contract must be a JSON object")
    required = {"request_id", "decision_type", "option_ids", "option_texts"}
    allowed = required | {"ordinal_values"}
    missing = required - value.keys()
    unknown = value.keys() - allowed
    if missing or unknown:
        raise ValueError(
            f"decision contract keys invalid (missing={sorted(missing)}, "
            f"unknown={sorted(unknown)})"
        )
    request_id = value["request_id"]
    decision_type = value["decision_type"]
    if not isinstance(request_id, str) or not isinstance(decision_type, str):
        raise ValueError("request_id and decision_type must be strings")
    ordinal_values = value.get("ordinal_values")
    if ordinal_values is not None and (
        not isinstance(ordinal_values, list)
        or any(
            isinstance(item, bool) or not isinstance(item, (int, float))
            for item in ordinal_values
        )
    ):
        raise ValueError("ordinal_values must be a JSON array of numbers")
    return TypedDecisionSpec(
        request_id=request_id,
        decision_type=decision_type,
        option_ids=_string_tuple(value["option_ids"], "option_ids"),
        option_texts=_string_tuple(value["option_texts"], "option_texts"),
        ordinal_values=tuple(ordinal_values) if ordinal_values is not None else None,
    )


def parse_request(value: Any) -> tuple[str, dict[str, Any], list[TypedDecisionSpec]]:
    """Validate a JSONL request and bind its typed output contract."""

    if not isinstance(value, dict):
        raise ValueError("request must be a JSON object")
    task = value.get("task")
    payload = value.get("payload")
    if not isinstance(task, str) or task not in {"text", "image", "audio", "video"}:
        raise ValueError("task must be one of text, image, audio, video")
    if not isinstance(payload, dict):
        raise ValueError("payload must be a JSON object")
    if task == "audio":
        if set(value) != {"task", "payload", "decisions"}:
            raise ValueError("audio request keys must be task, payload, and decisions")
        raw_specs = value["decisions"]
        if not isinstance(raw_specs, list) or not raw_specs:
            raise ValueError("audio decisions must be a nonempty JSON array")
    else:
        if set(value) != {"task", "payload", "decision"}:
            raise ValueError("request keys must be task, payload, and decision")
        raw_specs = [value["decision"]]
    return task, payload, [_decision_spec(item) for item in raw_specs]


def check_parameter_budget(manifest: dict[str, Any], maximum: int) -> int:
    """Return the verified package parameter count or fail closed over budget."""

    if isinstance(maximum, bool) or maximum < 1:
        raise ValueError("max-parameters must be a positive integer")
    try:
        count = int(manifest["unique_runtime_parameters"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("assembly manifest has no valid unique parameter count") from error
    if count < 1:
        raise ValueError("assembly manifest unique parameter count must be positive")
    if count > maximum:
        raise ValueError(
            f"runtime has {count} unique parameters, exceeding the configured limit {maximum}"
        )
    return count




class SessionFeatureStateManager:
    """Expose bounded, in-memory causal feature history over the JSONL protocol.

    The runtime hash is supplied by the loaded package manifest. Sessions are
    process-local and are never persisted; callers must provide a clock identity
    and monotonic evidence-availability timestamps.
    """

    def __init__(
        self,
        *,
        runtime_sha256: str,
        max_sessions: int = 8,
        max_session_bytes: int = 1_048_576,
    ) -> None:
        if type(max_sessions) is not int or max_sessions < 1:
            raise ValueError("max_sessions must be a positive integer")
        if type(max_session_bytes) is not int or max_session_bytes < 1:
            raise ValueError("max_session_bytes must be a positive integer")
        if (not isinstance(runtime_sha256, str) or len(runtime_sha256) != 64
                or any(character not in "0123456789abcdef" for character in runtime_sha256)):
            raise ValueError("runtime_sha256 must be a lowercase SHA-256 digest")
        self.runtime_sha256 = runtime_sha256
        self.max_sessions = max_sessions
        self.max_session_bytes = max_session_bytes
        self._sessions: dict[str, SessionTemporalFeatureState] = {}

    @staticmethod
    def _required_keys(request: dict[str, Any], required: set[str]) -> None:
        if request.keys() != required:
            raise ValueError(
                f"session request keys invalid (missing={sorted(required - request.keys())}, "
                f"unknown={sorted(request.keys() - required)})"
            )

    @staticmethod
    def _identifier(value: Any, name: str) -> str:
        if not isinstance(value, str) or not value.strip() or len(value) > 128:
            raise ValueError(f"{name} must be a non-empty string of at most 128 characters")
        return value

    @staticmethod
    def _finite_number(value: Any, name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be finite")
        try:
            result = float(value)
        except (OverflowError, ValueError) as error:
            raise ValueError(f"{name} must be finite") from error
        if not math.isfinite(result):
            raise ValueError(f"{name} must be finite")
        return result

    def _state(
        self, session_id: str, clock_id: str, *, create: bool
    ) -> SessionTemporalFeatureState:
        state = self._sessions.get(session_id)
        if state is None:
            if not create:
                raise ValueError("unknown session")
            if len(self._sessions) >= self.max_sessions:
                raise ValueError("session limit reached; close a session before opening another")
            state = SessionTemporalFeatureState(
                runtime_sha256=self.runtime_sha256,
                clock_id=clock_id,
                max_serialized_bytes=self.max_session_bytes,
                max_events_per_stream=8,
                max_streams=16,
            )
            self._sessions[session_id] = state
        return state

    @staticmethod
    def _feature_key(value: Any) -> ObservationFeatureKey:
        if not isinstance(value, dict):
            raise ValueError("feature_key must be a JSON object")
        expected = {
            "schema_version", "modality", "source_id", "source_revision",
            "observation_sha256", "encoder_id", "encoder_revision",
            "encoder_weights_sha256", "preprocessor_id", "preprocessor_revision",
            "preprocessing_sha256", "feature_name", "feature_dtype", "feature_shape",
            "temporal_policy", "start_ms", "end_ms",
        }
        if (set(value) != expected or type(value["schema_version"]) is not int
                or value["schema_version"] != 1):
            raise ValueError("feature_key schema is invalid")
        shape = value["feature_shape"]
        if (not isinstance(shape, list) or not shape
                or any(type(dimension) is not int or dimension < 1 for dimension in shape)):
            raise ValueError("feature_shape must contain positive integer dimensions")
        if math.prod(shape) > 65_536:
            raise ValueError("feature_shape exceeds the JSONL feature limit")
        fields = {key: item for key, item in value.items() if key != "schema_version"}
        fields["feature_shape"] = tuple(shape)
        try:
            key = ObservationFeatureKey(**fields)
        except (AttributeError, FeatureCacheError, TypeError, ValueError) as error:
            raise ValueError(f"feature_key is invalid: {error}") from error
        if key.feature_dtype != "float32":
            raise ValueError("session feature payloads must use float32")
        return key

    def handle(self, request: Any) -> dict[str, Any]:
        if not isinstance(request, dict):
            raise ValueError("session request must be a JSON object")
        operation = request.get("operation")
        if operation == "close_session":
            self._required_keys(request, {"operation", "session_id"})
            session_id = self._identifier(request["session_id"], "session_id")
            state = self._sessions.pop(session_id, None)
            return {
                "schema_version": 1,
                "operation": operation,
                "session_id": session_id,
                "closed": state is not None,
                "events_discarded": state.event_count if state else 0,
                "bytes_discarded": state.bytes_used if state else 0,
            }

        if operation in {"observe", "record_missing"}:
            required = {
                "operation", "session_id", "clock_id", "modality",
                "source_id", "available_at_ms",
            }
            if operation == "observe":
                required |= {"feature_key", "feature_payload_base64"}
            self._required_keys(request, required)
            session_id = self._identifier(request["session_id"], "session_id")
            clock_id = self._identifier(request["clock_id"], "clock_id")
            modality = request["modality"]
            if not isinstance(modality, str) or modality not in OBSERVATION_MODALITIES:
                raise ValueError("modality is unsupported")
            source_id = self._identifier(request["source_id"], "source_id")
            available_at_ms = self._finite_number(
                request["available_at_ms"], "available_at_ms"
            )
            key = None
            payload = None
            if operation == "observe":
                key = self._feature_key(request["feature_key"])
                if key.modality != modality or key.source_id != source_id:
                    raise ValueError("feature_key identity does not match the observation")
                encoded = request["feature_payload_base64"]
                if not isinstance(encoded, str) or len(encoded) > 350_000:
                    raise ValueError("feature_payload_base64 exceeds the input limit")
                try:
                    payload = base64.b64decode(encoded, validate=True)
                except (ValueError, TypeError) as error:
                    raise ValueError("feature_payload_base64 is invalid") from error
                if len(payload) != math.prod(key.feature_shape) * 4:
                    raise ValueError("feature payload byte length does not match feature_shape")
                if not all(
                    math.isfinite(value)
                    for (value,) in struct.iter_unpack("<f", payload)
                ):
                    raise ValueError("feature payload contains nonfinite values")
            state = self._state(session_id, clock_id, create=True)
            try:
                if operation == "record_missing":
                    state.record_missing(
                        modality=modality,
                        source_id=source_id,
                        available_at_ms=available_at_ms,
                        runtime_sha256=self.runtime_sha256,
                        clock_id=clock_id,
                    )
                    status = "recorded_missing"
                else:
                    state.update_feature(
                        modality=modality,
                        source_id=source_id,
                        available_at_ms=available_at_ms,
                        key=key,
                        payload=payload,
                        runtime_sha256=self.runtime_sha256,
                        clock_id=clock_id,
                    )
                    status = "stored"
            except TemporalFeatureStateError as error:
                raise ValueError(str(error)) from error
            return {
                "schema_version": 1,
                "operation": operation,
                "session_id": session_id,
                "status": status,
                "event_count": state.event_count,
                "bytes_used": state.bytes_used,
                "evictions": state.eviction_count,
            }

        if operation == "snapshot":
            self._required_keys(
                request,
                {
                    "operation", "session_id", "clock_id", "as_of_ms",
                    "max_age_ms", "required_streams",
                },
            )
            session_id = self._identifier(request["session_id"], "session_id")
            clock_id = self._identifier(request["clock_id"], "clock_id")
            required = request["required_streams"]
            if not isinstance(required, list) or not required:
                raise ValueError("required_streams must be a non-empty JSON array")
            streams: list[tuple[str, str]] = []
            for item in required:
                if not isinstance(item, dict) or set(item) != {"modality", "source_id"}:
                    raise ValueError("each required stream must have modality and source_id")
                modality = item["modality"]
                if not isinstance(modality, str) or modality not in OBSERVATION_MODALITIES:
                    raise ValueError("required stream modality is unsupported")
                streams.append((modality, self._identifier(item["source_id"], "source_id")))
            state = self._state(session_id, clock_id, create=False)
            try:
                snapshots = state.snapshot(
                    as_of_ms=self._finite_number(request["as_of_ms"], "as_of_ms"),
                    max_age_ms=self._finite_number(request["max_age_ms"], "max_age_ms"),
                    required_streams=streams,
                    runtime_sha256=self.runtime_sha256,
                    clock_id=clock_id,
                )
            except TemporalFeatureStateError as error:
                raise ValueError(str(error)) from error
            features = []
            for modality, source_id in streams:
                item = snapshots[modality, source_id]
                feature = {
                    "modality": modality,
                    "source_id": source_id,
                    "available": item.available,
                    "available_at_ms": item.available_at_ms,
                    "age_ms": item.age_ms,
                    "missing_reason": item.missing_reason,
                }
                if item.available:
                    feature["feature_key"] = item.feature_key.to_dict()
                    feature["payload_sha256"] = item.payload_sha256
                    feature["feature_payload_base64"] = base64.b64encode(
                        item.payload
                    ).decode("ascii")
                features.append(feature)
            return {
                "schema_version": 1,
                "operation": operation,
                "session_id": session_id,
                "features": features,
                "event_count": state.event_count,
                "bytes_used": state.bytes_used,
                "evictions": state.eviction_count,
            }

        raise ValueError("operation must be observe, record_missing, snapshot, or close_session")

    def limits_metadata(self) -> dict[str, int]:
        return {
            "max_sessions": self.max_sessions,
            "max_session_serialized_bytes": self.max_session_bytes,
            "max_events_per_stream": 8,
            "max_streams_per_session": 16,
        }


def run_jsonl(
    source: Iterable[str],
    sink: TextIO,
    dispatcher: Any,
    session_state: SessionFeatureStateManager | None = None,
) -> None:
    """Dispatch each nonblank JSONL request and emit only typed decision JSON."""

    for line_number, line in enumerate(source, start=1):
        if not line.strip():
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON on line {line_number}: {error.msg}") from error
        if isinstance(request, dict) and "operation" in request:
            if session_state is None:
                raise ValueError("temporal session operations are unavailable")
            record = session_state.handle(request)
            sink.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            sink.flush()
            continue
        task, payload, specs = parse_request(request)
        result = dispatcher.dispatch_typed(task, payload, decision_specs=specs)
        record = result.get("decision_json")
        if record is None:
            raise RuntimeError(f"runtime returned no typed decision JSON for {task}")
        sink.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        sink.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assembly", type=Path, required=True)
    parser.add_argument("--videoprism-python", type=Path, required=True)
    parser.add_argument("--videoprism-source", type=Path, required=True)
    parser.add_argument("--worker-log-dir", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, required=True)
    parser.add_argument("--max-parameters", type=int, default=200_000_000)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()

    from .runtime_dispatch import PathARuntimeDispatcher

    try:
        metadata_path = args.metadata_output.resolve()
        if metadata_path.exists():
            raise FileExistsError(f"refusing to overwrite runtime metadata: {metadata_path}")
        dispatcher = PathARuntimeDispatcher(
            args.assembly,
            videoprism_python=args.videoprism_python,
            videoprism_source=args.videoprism_source,
            worker_log_dir=args.worker_log_dir,
            device=args.device,
        )
        with dispatcher:
            unique_parameters = check_parameter_budget(
                dispatcher.manifest, args.max_parameters
            )
            assembly_manifest_sha256 = hashlib.sha256(
                (args.assembly / "package-manifest.json").read_bytes()
            ).hexdigest()
            session_state = SessionFeatureStateManager(
                runtime_sha256=assembly_manifest_sha256
            )
            run_jsonl(sys.stdin, sys.stdout, dispatcher, session_state=session_state)
            metadata = {
                "schema_version": 1,
                "device": args.device,
                "assembly_manifest_sha256": assembly_manifest_sha256,
                "temporal_session_limits": session_state.limits_metadata(),
                "unique_runtime_parameters": unique_parameters,
                "package_file_bytes_excluding_manifest": int(
                    dispatcher.manifest["package_file_bytes_excluding_manifest"]
                ),
                "package_target_met": bool(dispatcher.manifest["package_target_met"]),
                "workers": dispatcher.worker_metadata,
            }
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            with metadata_path.open("x", encoding="utf-8", newline="\n") as stream:
                json.dump(metadata, stream, indent=2, ensure_ascii=False, sort_keys=True)
                stream.write("\n")
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        print(f"path-a-decision-json: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
