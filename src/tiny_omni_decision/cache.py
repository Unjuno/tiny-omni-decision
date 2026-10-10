"""Immutable, content-verified cache for reusable observation features."""

from __future__ import annotations

import hashlib
import json
import os
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_HEADER_LENGTH = struct.Struct(">Q")


class FeatureCacheError(ValueError):
    """Raised when a feature-cache entry is malformed or does not match its key."""


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


@dataclass(frozen=True)
class ObservationFeatureKey:
    """Identity of an encoded observation, deliberately independent of any question."""

    modality: str
    source_id: str
    source_revision: str
    observation_sha256: str
    encoder_id: str
    encoder_revision: str
    encoder_weights_sha256: str
    preprocessor_id: str
    preprocessor_revision: str
    preprocessing_sha256: str
    feature_name: str
    feature_dtype: str
    feature_shape: tuple[int, ...]
    temporal_policy: str | None = None
    start_ms: int | None = None
    end_ms: int | None = None

    def __post_init__(self) -> None:
        if self.modality not in {"text", "image", "audio", "video", "sensor"}:
            raise FeatureCacheError(f"unsupported observation modality: {self.modality}")
        for name in (
            "source_id",
            "source_revision",
            "encoder_id",
            "encoder_revision",
            "preprocessor_id",
            "preprocessor_revision",
            "feature_name",
            "feature_dtype",
        ):
            if not getattr(self, name).strip():
                raise FeatureCacheError(f"{name} must be non-empty")
        for name in (
            "observation_sha256",
            "encoder_weights_sha256",
            "preprocessing_sha256",
        ):
            value = getattr(self, name)
            if not _is_sha256(value):
                raise FeatureCacheError(f"{name} must be a lowercase SHA-256 digest")
        if not self.feature_shape or any(dimension < 1 for dimension in self.feature_shape):
            raise FeatureCacheError("feature_shape must contain positive dimensions")
        if (self.start_ms is None) != (self.end_ms is None):
            raise FeatureCacheError("start_ms and end_ms must be set together")
        if self.start_ms is not None and self.end_ms is not None:
            if self.start_ms < 0 or self.end_ms <= self.start_ms:
                raise FeatureCacheError("time interval must be non-negative and increasing")
        if self.modality == "video" and not self.temporal_policy:
            raise FeatureCacheError("video features require a temporal sampling policy")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "modality": self.modality,
            "source_id": self.source_id,
            "source_revision": self.source_revision,
            "observation_sha256": self.observation_sha256,
            "encoder_id": self.encoder_id,
            "encoder_revision": self.encoder_revision,
            "encoder_weights_sha256": self.encoder_weights_sha256,
            "preprocessor_id": self.preprocessor_id,
            "preprocessor_revision": self.preprocessor_revision,
            "preprocessing_sha256": self.preprocessing_sha256,
            "feature_name": self.feature_name,
            "feature_dtype": self.feature_dtype,
            "feature_shape": list(self.feature_shape),
            "temporal_policy": self.temporal_policy,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
        }

    @property
    def cache_id(self) -> str:
        canonical = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()


@dataclass(frozen=True)
class CachedObservationFeature:
    key: ObservationFeatureKey
    payload: bytes
    payload_sha256: str
    entry_bytes: int


class ObservationFeatureCache:
    """Store exact feature payloads under complete, immutable observation identities."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _path(self, key: ObservationFeatureKey) -> Path:
        return self.root / f"{key.cache_id}.feature"

    def put(self, key: ObservationFeatureKey, payload: bytes) -> CachedObservationFeature:
        if not isinstance(payload, bytes) or not payload:
            raise FeatureCacheError("feature payload must be non-empty bytes")
        path = self._path(key)
        if path.exists():
            cached = self.get(key)
            if cached.payload != payload:
                raise FeatureCacheError("cache key already exists with a different feature payload")
            return cached

        payload_hash = hashlib.sha256(payload).hexdigest()
        metadata = {
            "key": key.to_dict(),
            "payload_sha256": payload_hash,
            "payload_bytes": len(payload),
        }
        encoded_metadata = json.dumps(
            metadata, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        entry = _HEADER_LENGTH.pack(len(encoded_metadata)) + encoded_metadata + payload
        self.root.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.root, delete=False) as temporary:
                temporary_path = Path(temporary.name)
                temporary.write(entry)
                temporary.flush()
                os.fsync(temporary.fileno())
            try:
                os.link(temporary_path, path)
            except FileExistsError as error:
                cached = self.get(key)
                if cached.payload != payload:
                    raise FeatureCacheError(
                        "concurrent writer stored a different payload for the same cache key"
                    ) from error
                return cached
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        return self.get(key)

    def get(self, key: ObservationFeatureKey) -> CachedObservationFeature:
        path = self._path(key)
        try:
            entry = path.read_bytes()
        except FileNotFoundError:
            raise FileNotFoundError(f"feature cache entry not found: {key.cache_id}") from None
        if len(entry) < _HEADER_LENGTH.size:
            raise FeatureCacheError("feature cache entry is truncated before metadata header")
        (metadata_length,) = _HEADER_LENGTH.unpack_from(entry)
        metadata_start = _HEADER_LENGTH.size
        metadata_end = metadata_start + metadata_length
        if metadata_length < 2 or metadata_end > len(entry):
            raise FeatureCacheError("feature cache metadata length is invalid")
        try:
            metadata = json.loads(entry[metadata_start:metadata_end].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise FeatureCacheError("feature cache metadata is not valid JSON") from error
        if metadata.get("key") != key.to_dict():
            raise FeatureCacheError("feature cache entry identity does not match requested key")
        payload = entry[metadata_end:]
        if len(payload) != metadata.get("payload_bytes"):
            raise FeatureCacheError("feature cache payload byte count mismatch")
        payload_hash = hashlib.sha256(payload).hexdigest()
        if payload_hash != metadata.get("payload_sha256"):
            raise FeatureCacheError("feature cache payload SHA-256 mismatch")
        return CachedObservationFeature(
            key=key,
            payload=payload,
            payload_sha256=payload_hash,
            entry_bytes=len(entry),
        )
