"""Immutable, packed storage for observation feature-cache entries.

The source cache remains the writable canonical form. Packed caches are
read-only snapshots with a sequential payload file and a compact SQLite index.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import struct
import tempfile
from pathlib import Path
from threading import RLock
from typing import Any

from tiny_omni_decision.cache import (
    CachedObservationFeature,
    FeatureCacheError,
    ObservationFeatureKey,
)

_HEADER_LENGTH = struct.Struct(">Q")
_PACK_NAME = "features.pack"
_INDEX_NAME = "index.sqlite3"
_MANIFEST_NAME = "manifest.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _decode_entry(
    key: ObservationFeatureKey, entry: bytes
) -> CachedObservationFeature:
    if len(entry) < _HEADER_LENGTH.size:
        raise FeatureCacheError("packed feature entry is truncated before metadata header")
    (metadata_length,) = _HEADER_LENGTH.unpack_from(entry)
    metadata_start = _HEADER_LENGTH.size
    metadata_end = metadata_start + metadata_length
    if metadata_length < 2 or metadata_end > len(entry):
        raise FeatureCacheError("packed feature metadata length is invalid")
    try:
        metadata = json.loads(entry[metadata_start:metadata_end].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FeatureCacheError("packed feature metadata is not valid JSON") from error
    if metadata.get("key") != key.to_dict():
        raise FeatureCacheError("packed feature identity does not match requested key")
    payload = entry[metadata_end:]
    if len(payload) != metadata.get("payload_bytes"):
        raise FeatureCacheError("packed feature payload byte count mismatch")
    payload_hash = hashlib.sha256(payload).hexdigest()
    if payload_hash != metadata.get("payload_sha256"):
        raise FeatureCacheError("packed feature payload SHA-256 mismatch")
    return CachedObservationFeature(
        key=key,
        payload=payload,
        payload_sha256=payload_hash,
        entry_bytes=len(entry),
    )


def _entry_identity(path: Path, entry: bytes) -> tuple[str, int, str]:
    if len(entry) < _HEADER_LENGTH.size:
        raise FeatureCacheError(f"source cache entry is truncated: {path.name}")
    (metadata_length,) = _HEADER_LENGTH.unpack_from(entry)
    metadata_start = _HEADER_LENGTH.size
    metadata_end = metadata_start + metadata_length
    if metadata_length < 2 or metadata_end > len(entry):
        raise FeatureCacheError(f"source cache metadata is invalid: {path.name}")
    try:
        metadata = json.loads(entry[metadata_start:metadata_end].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FeatureCacheError(f"source cache metadata is invalid: {path.name}") from error
    key_dict = metadata.get("key")
    if not isinstance(key_dict, dict):
        raise FeatureCacheError(f"source cache key is missing: {path.name}")
    key_bytes = json.dumps(
        key_dict, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    cache_id = hashlib.sha256(key_bytes).hexdigest()
    if path.stem != cache_id:
        raise FeatureCacheError(f"source cache filename does not match its key: {path.name}")
    payload = entry[metadata_end:]
    if len(payload) != metadata.get("payload_bytes"):
        raise FeatureCacheError(f"source cache payload length is invalid: {path.name}")
    payload_hash = hashlib.sha256(payload).hexdigest()
    if payload_hash != metadata.get("payload_sha256"):
        raise FeatureCacheError(f"source cache payload hash is invalid: {path.name}")
    return cache_id, len(entry), hashlib.sha256(entry).hexdigest()


class PackedObservationFeatureCache:
    """Build or read an immutable packed snapshot of an observation cache."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        try:
            manifest = json.loads((self.root / _MANIFEST_NAME).read_text(encoding="utf-8"))
        except (FileNotFoundError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise FeatureCacheError("packed cache manifest is missing or invalid") from error
        if manifest.get("schema_version") != 1:
            raise FeatureCacheError("unsupported packed cache schema version")
        self.manifest: dict[str, Any] = manifest
        self._lock = RLock()
        index_path = self.root / _INDEX_NAME
        self._connection = sqlite3.connect(
            f"file:{index_path.as_posix()}?mode=ro", uri=True, check_same_thread=False
        )
        self._pack_stream = (self.root / _PACK_NAME).open("rb")

    @classmethod
    def pack(cls, source_root: Path, target_root: Path) -> PackedObservationFeatureCache:
        """Pack source entries into a new target directory without overwriting it."""
        source_root = Path(source_root)
        target_root = Path(target_root)
        if not source_root.is_dir():
            raise FileNotFoundError(f"source feature-cache directory not found: {source_root}")
        if target_root.exists():
            raise FileExistsError(f"refusing to overwrite packed cache: {target_root}")
        target_root.parent.mkdir(parents=True, exist_ok=True)
        entries = sorted(source_root.glob("*.feature"))
        if not entries:
            raise FeatureCacheError("cannot pack an empty feature cache")

        stage = Path(tempfile.mkdtemp(prefix=f".{target_root.name}.", dir=target_root.parent))
        try:
            pack_path = stage / _PACK_NAME
            index_path = stage / _INDEX_NAME
            offset = 0
            with pack_path.open("xb") as pack_stream:
                connection = sqlite3.connect(index_path)
                try:
                    connection.execute(
                        "CREATE TABLE feature ("
                        "cache_id TEXT PRIMARY KEY, offset INTEGER NOT NULL, "
                        "length INTEGER NOT NULL, entry_sha256 TEXT NOT NULL) WITHOUT ROWID"
                    )
                    with connection:
                        for source_path in entries:
                            entry = source_path.read_bytes()
                            cache_id, entry_length, entry_hash = _entry_identity(
                                source_path, entry
                            )
                            connection.execute(
                                "INSERT INTO feature VALUES (?, ?, ?, ?)",
                                (cache_id, offset, entry_length, entry_hash),
                            )
                            pack_stream.write(entry)
                            offset += entry_length
                    integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
                    if integrity != "ok":
                        raise FeatureCacheError("packed cache index failed integrity check")
                finally:
                    connection.close()
                pack_stream.flush()
                os.fsync(pack_stream.fileno())

            manifest = {
                "schema_version": 1,
                "entry_count": len(entries),
                "pack_bytes": pack_path.stat().st_size,
                "pack_sha256": _sha256_file(pack_path),
                "index_bytes": index_path.stat().st_size,
                "index_sha256": _sha256_file(index_path),
            }
            manifest_path = stage / _MANIFEST_NAME
            manifest_path.write_text(
                json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
            )
            with manifest_path.open("r+b") as stream:
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(stage, target_root)
        except BaseException:
            shutil.rmtree(stage, ignore_errors=True)
            raise
        return cls(target_root)

    def verify(self) -> None:
        """Verify whole-file hashes and the index structure before distribution."""
        pack_path = self.root / _PACK_NAME
        index_path = self.root / _INDEX_NAME
        if pack_path.stat().st_size != self.manifest.get("pack_bytes"):
            raise FeatureCacheError("packed feature data size does not match manifest")
        if _sha256_file(pack_path) != self.manifest.get("pack_sha256"):
            raise FeatureCacheError("packed feature data SHA-256 does not match manifest")
        if index_path.stat().st_size != self.manifest.get("index_bytes"):
            raise FeatureCacheError("packed cache index size does not match manifest")
        if _sha256_file(index_path) != self.manifest.get("index_sha256"):
            raise FeatureCacheError("packed cache index SHA-256 does not match manifest")
        connection = sqlite3.connect(f"file:{index_path.as_posix()}?mode=ro", uri=True)
        try:
            integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
            count = connection.execute("SELECT COUNT(*) FROM feature").fetchone()[0]
        finally:
            connection.close()
        if integrity != "ok" or count != self.manifest.get("entry_count"):
            raise FeatureCacheError("packed cache index failed integrity or entry-count check")

    def get(self, key: ObservationFeatureKey) -> CachedObservationFeature:
        with self._lock:
            try:
                row = self._connection.execute(
                    "SELECT offset, length, entry_sha256 FROM feature WHERE cache_id = ?",
                    (key.cache_id,),
                ).fetchone()
            except sqlite3.Error as error:
                raise FeatureCacheError("packed cache index could not be read") from error
            if row is None:
                raise FileNotFoundError(f"packed feature cache entry not found: {key.cache_id}")
            offset, length, expected_hash = row
            self._pack_stream.seek(offset)
            entry = self._pack_stream.read(length)
        if len(entry) != length or hashlib.sha256(entry).hexdigest() != expected_hash:
            raise FeatureCacheError("packed feature entry is truncated or has changed")
        return _decode_entry(key, entry)

    def close(self) -> None:
        """Release the read-only index and pack handles."""
        with self._lock:
            self._pack_stream.close()
            self._connection.close()

    def __enter__(self) -> PackedObservationFeatureCache:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
