from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable

HEX64 = re.compile(r"[0-9a-f]{64}")
REQUIRED_FIELDS = {
    "sample_id",
    "ordered_options",
    "option_labels",
    "option_token_ids",
    "teacher_option_logits",
    "target_index",
    "temperature",
    "preprocessing_sha256",
    "model_sha256",
    "adapter_sha256",
    "processor_revision",
    "sampled_frame_policy",
    "source_media_sha256",
}
IDENTITY_FIELDS = tuple(sorted(REQUIRED_FIELDS - {"teacher_option_logits"}))


def _sha256(value: object, field: str) -> None:
    if not isinstance(value, str) or HEX64.fullmatch(value) is None:
        raise ValueError(f"{field}: expected lowercase SHA-256")


def _validate_record(record: dict[str, object]) -> None:
    if not isinstance(record, dict) or set(record) != REQUIRED_FIELDS:
        raise ValueError(f"cache record must contain exactly {sorted(REQUIRED_FIELDS)}")
    sample_id = record["sample_id"]
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("sample_id: nonempty string required")

    options = record["ordered_options"]
    labels = record["option_labels"]
    token_ids = record["option_token_ids"]
    logits = record["teacher_option_logits"]
    if not all(isinstance(value, list) for value in (options, labels, token_ids, logits)):
        raise ValueError("ordered option fields must be lists")
    option_count = len(options)
    if not 2 <= option_count <= 62:
        raise ValueError("ordered_options: expected 2..62 choices")
    if any(len(value) != option_count for value in (labels, token_ids, logits)):
        raise ValueError("ordered option fields must have equal length")
    if any(not isinstance(value, str) or not value for value in options):
        raise ValueError("ordered_options: nonempty strings required")
    if len(set(options)) != option_count:
        raise ValueError("ordered_options: duplicates are unsupported")
    if any(not isinstance(value, str) or not value for value in labels):
        raise ValueError("option_labels: nonempty strings required")
    if len(set(labels)) != option_count:
        raise ValueError("option_labels: labels must be distinct")
    if any(type(value) is not int for value in token_ids):
        raise ValueError("option_token_ids: integers required")
    if len(set(token_ids)) != option_count:
        raise ValueError("option_token_ids: token IDs must be distinct")
    if any(type(value) not in (int, float) or not math.isfinite(float(value)) for value in logits):
        raise ValueError("teacher_option_logits: finite numeric values required")

    target = record["target_index"]
    if type(target) is not int or not 0 <= target < option_count:
        raise ValueError("target_index: active option index required")
    temperature = record["temperature"]
    if type(temperature) not in (int, float) or not math.isfinite(float(temperature)):
        raise ValueError("temperature: finite numeric value required")
    if float(temperature) != 1.0:
        raise ValueError("temperature: cache v1 supports 1.0 only")

    for field in ("preprocessing_sha256", "model_sha256", "adapter_sha256"):
        _sha256(record[field], field)
    revision = record["processor_revision"]
    if not isinstance(revision, str) or not revision:
        raise ValueError("processor_revision: nonempty string required")

    frame_policy = record["sampled_frame_policy"]
    if not isinstance(frame_policy, dict):
        raise ValueError("sampled_frame_policy: object required")
    try:
        json.dumps(frame_policy, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("sampled_frame_policy: finite JSON values required") from exc

    media_hashes = record["source_media_sha256"]
    if not isinstance(media_hashes, list):
        raise ValueError("source_media_sha256: list required")
    for value in media_hashes:
        _sha256(value, "source_media_sha256")


def validate_teacher_cache_record(
    record: dict[str, object], expected: dict[str, object]
) -> None:
    """Reject stale/misaligned cache rows before their logits are consumed."""
    _validate_record(record)
    _validate_record(expected)
    for field in IDENTITY_FIELDS:
        if record[field] != expected[field]:
            raise ValueError(f"{field}: teacher cache identity mismatch")


def _canonical_line(record: dict[str, object]) -> bytes:
    _validate_record(record)
    return (
        json.dumps(
            record,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def cache_manifest_sha256(records: Iterable[dict[str, object]]) -> str:
    """Hash canonical rows in order; duplicate sample IDs are never accepted."""
    digest = hashlib.sha256(b"postquant-teacher-cache-v1\n")
    seen: set[str] = set()
    count = 0
    for record in records:
        _validate_record(record)
        sample_id = str(record["sample_id"])
        if sample_id in seen:
            raise ValueError(f"duplicate sample_id: {sample_id}")
        seen.add(sample_id)
        digest.update(_canonical_line(record))
        count += 1
    if count == 0:
        raise ValueError("teacher cache cannot be empty")
    return digest.hexdigest()


def sample_id_order_sha256(records: Iterable[dict[str, object]]) -> str:
    """Hash only ordered sample IDs for quick train/cache order comparison."""
    digest = hashlib.sha256(b"postquant-teacher-cache-sample-order-v1\n")
    seen: set[str] = set()
    count = 0
    for record in records:
        _validate_record(record)
        sample_id = str(record["sample_id"])
        if sample_id in seen:
            raise ValueError(f"duplicate sample_id: {sample_id}")
        seen.add(sample_id)
        digest.update((json.dumps(sample_id, ensure_ascii=True) + "\n").encode("utf-8"))
        count += 1
    if count == 0:
        raise ValueError("teacher cache cannot be empty")
    return digest.hexdigest()


def canonical_cache_bytes(
    records: Iterable[dict[str, object]],
) -> tuple[bytes, str, int]:
    rows = list(records)
    digest = cache_manifest_sha256(rows)
    return b"".join(_canonical_line(row) for row in rows), digest, len(rows)
