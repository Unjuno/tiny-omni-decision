"""Leakage-safe helpers for reading the Clevr-4 annotation split."""

from __future__ import annotations

import hashlib
import io
import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import BinaryIO


class _ByteReader:
    def __init__(self, stream: BinaryIO) -> None:
        self._stream = stream
        self._buffer = b""
        self._offset = 0

    def get(self) -> int | None:
        if self._offset >= len(self._buffer):
            self._buffer = self._stream.read(64 * 1024)
            self._offset = 0
            if not self._buffer:
                return None
        value = self._buffer[self._offset]
        self._offset += 1
        return value

    def skip_whitespace(self) -> int | None:
        value = self.get()
        while value in (9, 10, 13, 32):
            value = self.get()
        return value

    def read_string(self, first: int | None = None) -> tuple[str, bytes]:
        quote = self.get() if first is None else first
        if quote != ord('"'):
            raise ValueError("expected a JSON string")
        token = bytearray([quote])
        escaped = False
        while (value := self.get()) is not None:
            token.append(value)
            if escaped:
                escaped = False
            elif value == ord("\\"):
                escaped = True
            elif value == ord('"'):
                try:
                    decoded = json.loads(token.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ValueError("invalid JSON string token") from exc
                if not isinstance(decoded, str):
                    raise ValueError("expected a JSON string")
                return decoded, bytes(token)
        raise ValueError("unterminated JSON string")

    def read_compound_value(self, first: int) -> bytes:
        if first not in (ord("{"), ord("[")):
            raise ValueError("expected a JSON object or array")
        token = bytearray([first])
        depth = 1
        in_string = False
        escaped = False
        while (value := self.get()) is not None:
            token.append(value)
            if in_string:
                if escaped:
                    escaped = False
                elif value == ord("\\"):
                    escaped = True
                elif value == ord('"'):
                    in_string = False
            elif value == ord('"'):
                in_string = True
            elif value in (ord("{"), ord("[")):
                depth += 1
            elif value in (ord("}"), ord("]")):
                depth -= 1
                if depth == 0:
                    return bytes(token)
        raise ValueError("unterminated JSON compound value")

    def read_opaque_value(self, first: int) -> bytes:
        """Consume one JSON value without decoding its contents."""
        if first == ord('"'):
            _, token = self.read_string(first)
            return token
        if first in (ord("{"), ord("[")):
            return self.read_compound_value(first)
        token = bytearray([first])
        while (value := self.get()) is not None and value not in (
            ord(","),
            ord("}"),
            ord("]"),
            9,
            10,
            13,
            32,
        ):
            token.append(value)
        if value is not None:
            self._offset -= 1
        return bytes(token)


def _read_split_field(raw_object: bytes) -> str:
    reader = _ByteReader(io.BytesIO(raw_object))
    if reader.skip_whitespace() != ord("{"):
        raise ValueError("annotation value must be a JSON object")
    while True:
        first = reader.skip_whitespace()
        if first == ord("}"):
            break
        key, _ = reader.read_string(first)
        if reader.skip_whitespace() != ord(":"):
            raise ValueError("expected colon after annotation field")
        value_first = reader.skip_whitespace()
        if value_first is None:
            raise ValueError("annotation ended before a field value")
        if key == "split":
            if value_first != ord('"'):
                raise ValueError("split field must be a string")
            split, _ = reader.read_string(value_first)
            return split
        reader.read_opaque_value(value_first)
        separator = reader.skip_whitespace()
        if separator == ord("}"):
            break
        if separator != ord(","):
            raise ValueError("expected comma between annotation fields")
    raise ValueError("annotation object has no split field")


def read_training_annotations(stream: BinaryIO) -> tuple[list[dict[str, object]], dict[str, int]]:
    """Return train annotations and counts of split labels.

    Non-train object values are scanned only far enough to locate their
    top-level ``split`` string. Their label values are skipped as opaque JSON
    bytes and are never decoded into Python objects.
    """
    reader = _ByteReader(stream)
    if reader.skip_whitespace() != ord("{"):
        raise ValueError("annotation document must be a JSON object")

    train_rows: list[dict[str, object]] = []
    split_counts: Counter[str] = Counter()
    while True:
        first = reader.skip_whitespace()
        if first == ord("}"):
            break
        if first is None:
            raise ValueError("annotation document ended unexpectedly")
        image_id, _ = reader.read_string(first)
        if reader.skip_whitespace() != ord(":"):
            raise ValueError("expected colon after image identity")
        object_first = reader.skip_whitespace()
        if object_first != ord("{"):
            raise ValueError("annotation value must be a JSON object")
        raw_object = reader.read_compound_value(object_first)
        split = _read_split_field(raw_object)
        split_counts[split] += 1
        if split == "train":
            try:
                annotation = json.loads(raw_object)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid train annotation for {image_id}") from exc
            if annotation.get("split") != "train":
                raise ValueError("parsed train split changed during validation")
            required = {"color", "texture", "count", "shape", "split"}
            if not required <= annotation.keys():
                raise ValueError(f"train annotation missing fields for {image_id}")
            train_rows.append({"id": image_id, **annotation})

        separator = reader.skip_whitespace()
        if separator == ord("}"):
            break
        if separator != ord(","):
            raise ValueError("expected comma between image annotations")

    if reader.skip_whitespace() is not None:
        raise ValueError("unexpected trailing data after annotation document")
    return train_rows, dict(split_counts)


def read_annotations_for_split(
    stream: BinaryIO, *, requested_split: str
) -> tuple[list[dict[str, object]], dict[str, int]]:
    """Decode labels only for one explicitly requested official split.

    Other split payloads remain opaque while the top-level ``split`` field is
    inspected. This is intended for one-time evaluation of a named public
    split after the evaluated model and procedure are frozen.
    """
    if not requested_split or requested_split == "train":
        raise ValueError("requested_split must name a non-training split")

    reader = _ByteReader(stream)
    if reader.skip_whitespace() != ord("{"):
        raise ValueError("annotation document must be a JSON object")

    requested_rows: list[dict[str, object]] = []
    split_counts: Counter[str] = Counter()
    while True:
        first = reader.skip_whitespace()
        if first == ord("}"):
            break
        if first is None:
            raise ValueError("annotation document ended unexpectedly")
        image_id, _ = reader.read_string(first)
        if reader.skip_whitespace() != ord(":"):
            raise ValueError("expected colon after annotation identity")
        object_first = reader.skip_whitespace()
        if object_first != ord("{"):
            raise ValueError("annotation value must be a JSON object")
        raw_object = reader.read_compound_value(object_first)
        split = _read_split_field(raw_object)
        split_counts[split] += 1
        if split == requested_split:
            try:
                annotation = json.loads(raw_object)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid {requested_split} annotation for {image_id}") from exc
            if annotation.get("split") != requested_split:
                raise ValueError("parsed requested split changed during validation")
            requested_rows.append({"id": image_id, **annotation})

        separator = reader.skip_whitespace()
        if separator == ord("}"):
            break
        if separator != ord(","):
            raise ValueError("expected comma between image annotations")

    if reader.skip_whitespace() is not None:
        raise ValueError("unexpected trailing data after annotation document")
    return requested_rows, dict(split_counts)


def split_records_by_asset(
    records: Iterable[Mapping[str, object]],
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    """Split record IDs by underlying asset with a stable seeded hash order."""
    if not math.isfinite(validation_fraction) or not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    ids: set[str] = set()
    asset_to_ids: dict[str, list[str]] = {}
    for record in records:
        record_id = record.get("id")
        asset_id = record.get("asset_id")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError("every record needs a nonempty string id")
        if not isinstance(asset_id, str) or not asset_id:
            raise ValueError("every record needs a nonempty string asset_id")
        if record_id in ids:
            raise ValueError(f"duplicate record id: {record_id}")
        ids.add(record_id)
        asset_to_ids.setdefault(asset_id, []).append(record_id)
    assets = sorted(asset_to_ids)
    if len(assets) < 2:
        raise ValueError("at least two unique assets are needed for a split")
    ranked_assets = sorted(
        assets,
        key=lambda asset: hashlib.sha256(f"{seed}:{asset}".encode()).digest(),
    )
    validation_count = min(max(1, math.ceil(len(assets) * validation_fraction)), len(assets) - 1)
    validation_assets = set(ranked_assets[:validation_count])
    train_ids = sorted(
        record_id
        for asset, record_ids in asset_to_ids.items()
        if asset not in validation_assets
        for record_id in record_ids
    )
    validation_ids = sorted(
        record_id
        for asset, record_ids in asset_to_ids.items()
        if asset in validation_assets
        for record_id in record_ids
    )
    return train_ids, validation_ids


def require_disjoint_media(
    train_records: Iterable[Mapping[str, object]],
    validation_records: Iterable[Mapping[str, object]],
) -> None:
    train_media = _media_identities(train_records, "train")
    validation_media = _media_identities(validation_records, "validation")
    overlap = train_media & validation_media
    if overlap:
        raise ValueError(f"media identity overlap: {len(overlap)}")


def _media_identities(records: Iterable[Mapping[str, object]], split: str) -> set[str]:
    media: set[str] = set()
    for record in records:
        identity = record.get("media_sha256")
        if not isinstance(identity, str) or len(identity) != 64:
            raise ValueError(f"{split} record needs a SHA-256 media identity")
        try:
            bytes.fromhex(identity)
        except ValueError as exc:
            raise ValueError(f"{split} record has an invalid media SHA-256") from exc
        media.add(identity.lower())
    return media
