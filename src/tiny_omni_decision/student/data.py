"""Option-bound Teacher caches, split checks, and local-media integrity."""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROLES = {"train", "validation", "evaluation"}


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    """Exclusive creation; never replace a user artifact."""
    text = canonical(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text + "\n")
        stream.flush()
        os.fsync(stream.fileno())


@dataclass
class Record:
    sample_id: str
    group_id: str
    content_id: str
    modality: str
    inputs: dict[str, Any]
    options: list[str]
    target: int
    teacher_logits: list[float]
    media_hashes: dict[str, str]

    def __post_init__(self) -> None:
        identities = (self.sample_id, self.group_id, self.content_id)
        if not all(isinstance(v, str) and v for v in identities):
            raise ValueError("sample/group/content identities are required")
        if self.modality not in {"text", "image", "audio", "video"}:
            raise ValueError("invalid modality")
        if not isinstance(self.inputs, dict) or not isinstance(self.inputs.get("text"), str):
            raise ValueError("inputs require text")
        if set(self.inputs) - {"text", "image", "audio", "video"}:
            raise ValueError("unsupported input keys")
        if not 2 <= len(self.options) <= 20 or not all(
            isinstance(o, str) and o.strip() for o in self.options
        ) or len(set(self.options)) != len(self.options):
            raise ValueError("need 2..20 distinct nonempty options")
        if type(self.target) is not int or not 0 <= self.target < len(self.options):
            raise ValueError("target must be an option index")
        if len(self.teacher_logits) != len(self.options) or not all(
            type(v) in (int, float) and math.isfinite(v) for v in self.teacher_logits
        ):
            raise ValueError("Teacher logits must be finite and match the options")
        paths = []
        for kind in {"image", "audio", "video"} & self.inputs.keys():
            value = self.inputs[kind]
            values = value if isinstance(value, list) else [value]
            if not values or not all(isinstance(v, str) and v for v in values):
                raise ValueError("media inputs must be nonempty paths")
            paths.extend(values)
        if self.modality != "text" and self.modality not in self.inputs:
            raise ValueError("missing modality input")
        if set(paths) != set(self.media_hashes):
            raise ValueError("every media path requires a hash")
        if any(len(h) != 64 or any(c not in "0123456789abcdef" for c in h)
               for h in self.media_hashes.values()):
            raise ValueError("invalid media SHA-256")


@dataclass
class Cache:
    role: str
    teacher_id: str
    records: list[Record]
    source_sha256: str

    def __post_init__(self) -> None:
        if self.role not in ROLES or not self.teacher_id or not self.records:
            raise ValueError("cache requires a role, Teacher identity and records")
        ids = [r.sample_id for r in self.records]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate sample IDs")


def write_cache(
    path: Path, records: list[Record], *, role: str, teacher_id: str, source_sha256: str
) -> None:
    cache = Cache(role, teacher_id, records, source_sha256)
    lines = [canonical(asdict(record)) for record in cache.records]
    payload = "\n".join(lines) + "\n"
    header = {
        "format": "tiny-omni-option-cache-v1", "role": role, "teacher_id": teacher_id,
        "source_sha256": source_sha256, "count": len(records),
        "records_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(canonical(header) + "\n" + payload)
        stream.flush()
        os.fsync(stream.fileno())


def load_cache(path: Path, expected_role: str) -> Cache:
    with path.open(encoding="utf-8") as stream:
        header = json.loads(stream.readline())
        payload = stream.read()
    if header.get("format") != "tiny-omni-option-cache-v1":
        raise ValueError("unsupported cache format")
    if header.get("role") != expected_role:
        raise ValueError(f"cache role must be {expected_role}")
    if hashlib.sha256(payload.encode("utf-8")).hexdigest() != header["records_sha256"]:
        raise ValueError("cache digest mismatch: options, logits or inputs changed")
    records = [Record(**json.loads(line)) for line in payload.splitlines() if line.strip()]
    if len(records) != header["count"]:
        raise ValueError("cache count mismatch")
    return Cache(expected_role, header["teacher_id"], records, header["source_sha256"])


def assert_disjoint(left: Cache, right: Cache) -> None:
    if left.teacher_id != right.teacher_id:
        raise ValueError("Teacher identity differs across caches")
    for attribute in ("sample_id", "group_id", "content_id"):
        a = {getattr(r, attribute) for r in left.records}
        b = {getattr(r, attribute) for r in right.records}
        if a & b:
            raise ValueError(f"split overlap in {attribute}: {sorted(a & b)[:3]}")
    a = {h for r in left.records for h in r.media_hashes.values()}
    b = {h for r in right.records for h in r.media_hashes.values()}
    if a & b:
        raise ValueError("split overlap in media content")


def media_path(root: Path, value: str) -> Path:
    value = value.replace("\\", "/")
    if ":" in value or value.startswith("/") or ".." in value.split("/"):
        raise ValueError("media path must be relative, local and contained in data root")
    root = root.resolve()
    result = (root / value).resolve()
    if root not in result.parents:
        raise ValueError("media path escapes data root")
    if not result.is_file():
        raise FileNotFoundError(result)
    return result


def validate_media(cache: Cache, root: Path) -> None:
    seen: dict[str, str] = {}
    for record in cache.records:
        for relative, expected in record.media_hashes.items():
            if relative not in seen:
                seen[relative] = file_hash(media_path(root, relative))
            if seen[relative] != expected:
                raise ValueError(f"media checksum mismatch: {relative}")


def split_identities(*caches: Cache) -> dict:
    """Persist split exclusions so final evaluation cannot reuse selection data."""
    if not caches or len({c.teacher_id for c in caches}) != 1:
        raise ValueError("split guard requires one Teacher identity")
    result = {"teacher_id": caches[0].teacher_id}
    for attribute in ("sample_id", "group_id", "content_id"):
        result[attribute] = sorted({getattr(r, attribute) for c in caches for r in c.records})
    result["media_hashes"] = sorted({h for c in caches for r in c.records
                                    for h in r.media_hashes.values()})
    result["source_sha256"] = sorted({c.source_sha256 for c in caches})
    return result


def assert_heldout(cache: Cache, guard: dict) -> None:
    if cache.role != "evaluation" or cache.teacher_id != guard["teacher_id"]:
        raise ValueError("final evaluation requires matching Teacher and evaluation role")
    actual = split_identities(cache)
    for attribute in ("sample_id", "group_id", "content_id", "media_hashes", "source_sha256"):
        if set(actual[attribute]) & set(guard[attribute]):
            raise ValueError(f"final evaluation overlap in {attribute}")
