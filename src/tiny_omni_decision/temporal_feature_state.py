"""Bounded, causal feature history for repeated decisions within one session.

This stores caller-provided frozen features. It does not read media, align
clocks, predict dynamics, or update model parameters.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass
from threading import RLock

from tiny_omni_decision.cache import ObservationFeatureKey


class TemporalFeatureStateError(ValueError):
    """Raised when state updates violate identity, time, or resource contracts."""


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _nonempty(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise TemporalFeatureStateError(f"{name} must be non-empty")


@dataclass(frozen=True)
class _Event:
    modality: str
    source_id: str
    available_at_ms: float
    kind: str
    key: ObservationFeatureKey | None
    payload: bytes | None
    payload_sha256: str | None
    entry_bytes: int


@dataclass(frozen=True)
class TemporalFeatureSnapshot:
    modality: str
    source_id: str
    available: bool
    payload: bytes | None
    payload_sha256: str | None
    feature_key: ObservationFeatureKey | None
    available_at_ms: float | None
    age_ms: float | None
    missing_reason: str | None


class SessionTemporalFeatureState:
    """Keep a bounded time history of frozen features across session requests.

    Timestamps are evidence-availability times on the caller-verified common
    clock. Model/preprocessor compatibility is represented by the runtime hash;
    a clock or runtime change requires explicit ``rebind`` and clears history.
    Byte accounting includes canonical event metadata plus payload bytes, but
    excludes Python object/container overhead.
    """

    def __init__(
        self,
        *,
        runtime_sha256: str,
        clock_id: str,
        max_serialized_bytes: int,
        max_events_per_stream: int = 8,
        max_streams: int = 16,
    ) -> None:
        self._validate_context(runtime_sha256, clock_id)
        for value, name in (
            (max_serialized_bytes, "max_serialized_bytes"),
            (max_events_per_stream, "max_events_per_stream"),
            (max_streams, "max_streams"),
        ):
            if type(value) is not int or value < 1:
                raise TemporalFeatureStateError(f"{name} must be a positive integer")
        self.runtime_sha256 = runtime_sha256
        self.clock_id = clock_id
        self.max_serialized_bytes = max_serialized_bytes
        self.max_events_per_stream = max_events_per_stream
        self.max_streams = max_streams
        self._events: OrderedDict[tuple[str, str, float], _Event] = OrderedDict()
        self._by_stream: dict[tuple[str, str], deque[tuple[str, str, float]]] = defaultdict(deque)
        self._last_timestamp: dict[tuple[str, str], float] = {}
        self._stream_evictions: dict[tuple[str, str], int] = defaultdict(int)
        self._bytes_used = 0
        self._evictions = 0
        self._invalidations = 0
        self._lock = RLock()

    @staticmethod
    def _validate_context(runtime_sha256: str, clock_id: str) -> None:
        if not isinstance(runtime_sha256, str) or not _is_sha256(runtime_sha256):
            raise TemporalFeatureStateError("runtime hash must be a lowercase SHA-256 digest")
        _nonempty(clock_id, "clock_id")

    def _check_context(self, runtime_sha256: str, clock_id: str) -> None:
        self._validate_context(runtime_sha256, clock_id)
        if runtime_sha256 != self.runtime_sha256:
            raise TemporalFeatureStateError("runtime hash does not match bound session state")
        if clock_id != self.clock_id:
            raise TemporalFeatureStateError("clock does not match bound session state")

    @staticmethod
    def _validate_stream(modality: str, source_id: str) -> tuple[str, str]:
        if modality not in {"text", "image", "audio", "video", "sensor"}:
            raise TemporalFeatureStateError(f"unsupported modality: {modality}")
        _nonempty(source_id, "source_id")
        return modality, source_id

    @staticmethod
    def _validate_time(timestamp: float) -> float:
        if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)):
            raise TemporalFeatureStateError("timestamp must be finite")
        timestamp = float(timestamp)
        if not math.isfinite(timestamp):
            raise TemporalFeatureStateError("timestamp must be finite")
        return timestamp

    @staticmethod
    def _entry_size(
        *,
        modality: str,
        source_id: str,
        available_at_ms: float,
        kind: str,
        key: ObservationFeatureKey | None,
        payload: bytes | None,
        payload_sha256: str | None,
    ) -> int:
        metadata = {
            "schema_version": 1,
            "modality": modality,
            "source_id": source_id,
            "available_at_ms": available_at_ms,
            "kind": kind,
            "feature_key": key.to_dict() if key is not None else None,
            "payload_sha256": payload_sha256,
            "payload_bytes": len(payload) if payload is not None else 0,
        }
        encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return len(encoded) + (len(payload) if payload is not None else 0)

    def _make_event(
        self,
        *,
        modality: str,
        source_id: str,
        available_at_ms: float,
        kind: str,
        key: ObservationFeatureKey | None,
        payload: bytes | None,
    ) -> _Event:
        timestamp = self._validate_time(available_at_ms)
        if kind == "feature":
            valid_payload = isinstance(payload, bytes) and bool(payload)
            if not isinstance(key, ObservationFeatureKey) or not valid_payload:
                raise TemporalFeatureStateError("feature update requires a key and non-empty bytes")
            if key.modality != modality or key.source_id != source_id:
                raise TemporalFeatureStateError("feature key identity does not match stream")
            payload_sha256 = hashlib.sha256(payload).hexdigest()
        else:
            if key is not None or payload is not None:
                raise TemporalFeatureStateError(
                    "missing update cannot carry a feature key or payload"
                )
            payload_sha256 = None
        entry_bytes = self._entry_size(
            modality=modality,
            source_id=source_id,
            available_at_ms=timestamp,
            kind=kind,
            key=key,
            payload=payload,
            payload_sha256=payload_sha256,
        )
        if entry_bytes > self.max_serialized_bytes:
            raise TemporalFeatureStateError("event exceeds state byte budget")
        return _Event(
            modality,
            source_id,
            timestamp,
            kind,
            key,
            payload,
            payload_sha256,
            entry_bytes,
        )

    def _remove(self, event_id: tuple[str, str, float]) -> None:
        event = self._events.pop(event_id)
        stream = (event.modality, event.source_id)
        self._bytes_used -= event.entry_bytes
        self._by_stream[stream].remove(event_id)
        self._stream_evictions[stream] += 1
        self._evictions += 1

    def _insert(
        self,
        event: _Event,
        *,
        runtime_sha256: str,
        clock_id: str,
    ) -> None:
        self._check_context(runtime_sha256, clock_id)
        stream = (event.modality, event.source_id)
        previous_time = self._last_timestamp.get(stream)
        if previous_time is not None and event.available_at_ms <= previous_time:
            raise TemporalFeatureStateError("per-stream timestamps must strictly increase")
        if stream not in self._by_stream and len(self._by_stream) >= self.max_streams:
            raise TemporalFeatureStateError("session stream limit reached")

        while len(self._by_stream[stream]) >= self.max_events_per_stream:
            self._remove(self._by_stream[stream][0])
        while self._bytes_used + event.entry_bytes > self.max_serialized_bytes:
            oldest_id = next(iter(self._events))
            self._remove(oldest_id)

        event_id = (event.modality, event.source_id, event.available_at_ms)
        self._events[event_id] = event
        self._by_stream[stream].append(event_id)
        self._last_timestamp[stream] = event.available_at_ms
        self._bytes_used += event.entry_bytes

    def update_feature(
        self,
        *,
        modality: str,
        source_id: str,
        available_at_ms: float,
        key: ObservationFeatureKey,
        payload: bytes,
        runtime_sha256: str,
        clock_id: str,
    ) -> None:
        """Append one feature event; late, duplicate, incompatible or oversized updates fail."""
        stream = self._validate_stream(modality, source_id)
        event = self._make_event(
            modality=stream[0],
            source_id=stream[1],
            available_at_ms=available_at_ms,
            kind="feature",
            key=key,
            payload=payload,
        )
        with self._lock:
            self._insert(event, runtime_sha256=runtime_sha256, clock_id=clock_id)

    def record_missing(
        self,
        *,
        modality: str,
        source_id: str,
        available_at_ms: float,
        runtime_sha256: str,
        clock_id: str,
    ) -> None:
        """Record explicit absence so a prior value is not silently reused."""
        stream = self._validate_stream(modality, source_id)
        event = self._make_event(
            modality=stream[0],
            source_id=stream[1],
            available_at_ms=available_at_ms,
            kind="missing",
            key=None,
            payload=None,
        )
        with self._lock:
            self._insert(event, runtime_sha256=runtime_sha256, clock_id=clock_id)

    def snapshot(
        self,
        *,
        as_of_ms: float,
        max_age_ms: float,
        required_streams: list[tuple[str, str]] | tuple[tuple[str, str], ...],
        runtime_sha256: str,
        clock_id: str,
    ) -> dict[tuple[str, str], TemporalFeatureSnapshot]:
        """Return the latest available, non-stale feature per required stream."""
        as_of = self._validate_time(as_of_ms)
        age_limit = self._validate_time(max_age_ms)
        if age_limit < 0:
            raise TemporalFeatureStateError("max_age_ms must be nonnegative")
        if not isinstance(required_streams, (list, tuple)) or not required_streams:
            raise TemporalFeatureStateError("required_streams must be a non-empty list or tuple")
        streams = [self._validate_stream(*stream) for stream in required_streams]
        if len(set(streams)) != len(streams):
            raise TemporalFeatureStateError("required streams must be unique")
        if len(streams) > self.max_streams:
            raise TemporalFeatureStateError("requested stream count exceeds session stream limit")

        result: dict[tuple[str, str], TemporalFeatureSnapshot] = {}
        with self._lock:
            self._check_context(runtime_sha256, clock_id)
            for stream in streams:
                event_ids = self._by_stream.get(stream, ())
                event = next(
                    (self._events[event_id] for event_id in reversed(event_ids)
                     if self._events[event_id].available_at_ms <= as_of),
                    None,
                )
                if event is None:
                    if not event_ids and stream not in self._last_timestamp:
                        reason = "never_observed"
                    elif self._stream_evictions.get(stream, 0):
                        reason = "history_evicted_or_not_observed"
                    else:
                        reason = "not_yet_available"
                    result[stream] = TemporalFeatureSnapshot(
                        *stream, False, None, None, None, None, None, reason
                    )
                    continue
                age = as_of - event.available_at_ms
                if event.kind == "missing":
                    result[stream] = TemporalFeatureSnapshot(
                        *stream, False, None, None, None, event.available_at_ms, age,
                        "explicitly_missing",
                    )
                elif age > age_limit:
                    result[stream] = TemporalFeatureSnapshot(
                        *stream, False, None, None, None, event.available_at_ms, age, "stale"
                    )
                else:
                    result[stream] = TemporalFeatureSnapshot(
                        *stream,
                        True,
                        event.payload,
                        event.payload_sha256,
                        event.key,
                        event.available_at_ms,
                        age,
                        None,
                    )
        return result

    def rebind(self, *, runtime_sha256: str, clock_id: str) -> int:
        """Explicitly invalidate all history before changing runtime or clock identity."""
        self._validate_context(runtime_sha256, clock_id)
        with self._lock:
            removed = len(self._events)
            self._events.clear()
            self._by_stream.clear()
            self._last_timestamp.clear()
            self._stream_evictions.clear()
            self._bytes_used = 0
            self._invalidations += removed
            self.runtime_sha256 = runtime_sha256
            self.clock_id = clock_id
            return removed

    @property
    def bytes_used(self) -> int:
        with self._lock:
            return self._bytes_used

    @property
    def event_count(self) -> int:
        with self._lock:
            return len(self._events)

    @property
    def eviction_count(self) -> int:
        with self._lock:
            return self._evictions

    @property
    def invalidation_count(self) -> int:
        with self._lock:
            return self._invalidations
