"""Audit previously exposed CLEVRER train scenes without opening media bytes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SOURCE_KEYS = {
    "source", "source_id", "dataset", "dataset_id", "dataset_name", "source_dataset",
    "media_source",
}
SCENE_KEYS = {
    "scene_index", "scene_id", "scene_group_id", "video_id", "video_filename", "source",
    "source_id",
}
IDENTITY_KEYS = SCENE_KEYS | {"id", "parent_id", "asset_id", "media_id"}
EXCLUDED_PARTS = {
    "test", "tests", "sealed", "audit", "__pycache__", ".git", "node_modules",
    ".venv", "venv", "env", "feature-env", "site-packages", "__pypackages__"
}
SCENE_RE = re.compile(r"(?:clevrer[^0-9]{0,20}|video[_-]?)(\d{1,5})", re.IGNORECASE)


def _stream_json_array(path: Path) -> Iterator[Any]:
    """Decode a JSON array incrementally, retaining at most one item at a time."""
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as stream:
        if stream.read(1) != "[":
            raise ValueError(f"expected JSON array: {path}")
        buffer = ""
        eof = False
        while True:
            buffer = buffer.lstrip()
            if buffer.startswith("]"):
                return
            if buffer.startswith(","):
                buffer = buffer[1:].lstrip()
            while not buffer and not eof:
                chunk = stream.read(1024 * 1024)
                eof = not chunk
                buffer += chunk
            if not buffer:
                raise ValueError(f"unterminated JSON array: {path}")
            try:
                value, end = decoder.raw_decode(buffer)
            except json.JSONDecodeError as error:
                if eof:
                    raise ValueError(f"invalid JSON array: {path}") from error
                buffer += stream.read(1024 * 1024)
                if not buffer:
                    eof = True
                continue
            yield value
            buffer = buffer[end:]


def iter_train_scene_ids(path: Path, *, expected_sha256: str) -> Iterator[int]:
    actual = sha256_file(path)
    if actual.lower() != expected_sha256.lower():
        raise ValueError(f"source SHA-256 mismatch for {path}: {actual}")
    seen: set[int] = set()
    for row in _stream_json_array(path):
        if not isinstance(row, dict) or "scene_index" not in row:
            raise ValueError(f"train record missing scene_index: {path}")
        scene = int(row["scene_index"])
        if not 0 <= scene < 10_000:
            raise ValueError(f"scene_index outside official train range: {scene}")
        if scene not in seen:
            seen.add(scene)
            yield scene


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _walk_identity_values(value: Any, key: str | None = None) -> Iterator[tuple[str | None, Any]]:
    if isinstance(value, dict):
        for child_key, child in value.items():
            yield from _walk_identity_values(child, str(child_key).lower())
    elif isinstance(value, list):
        for child in value:
            yield from _walk_identity_values(child, key)
    else:
        yield key, value


def exposed_scene_ids(rows: Iterable[dict[str, Any]]) -> set[int]:
    result: set[int] = set()
    for row in rows:
        values = list(_walk_identity_values(row))
        source_tagged = any(
            key in SOURCE_KEYS and isinstance(value, str) and "clevrer" in value.lower()
            for key, value in values
        )
        named_identity = any(
            key in IDENTITY_KEYS and isinstance(value, str) and "clevrer" in value.lower()
            for key, value in values
        )
        if not (source_tagged or named_identity):
            continue
        for key, value in values:
            if (
                key in IDENTITY_KEYS
                and isinstance(value, (str, int))
                and not isinstance(value, bool)
            ):
                text = str(value)
                match = SCENE_RE.search(text) if isinstance(value, str) else None
                if match and (source_tagged or "clevrer" in text.lower()):
                    scene = int(match.group(1))
                    if 0 <= scene < 10_000:
                        result.add(scene)
                elif key in SCENE_KEYS and isinstance(value, int):
                    if 0 <= value < 10_000:
                        result.add(value)
    return result


def candidate_scene_ids(
    official: Iterable[int], *, exposed: set[int], cached_media: set[int]
) -> list[int]:
    return sorted(set(official) - exposed - cached_media)


def sha256_ids(scene_ids: Iterable[int]) -> str:
    payload = ",".join(str(value) for value in sorted(set(scene_ids))).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def write_receipt(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _excluded(path: Path) -> bool:
    if any(part.lower() in EXCLUDED_PARTS for part in path.parts[:-1]):
        return True
    filename_tokens = set(re.split(r"[^a-z0-9]+", path.name.lower()))
    return bool(filename_tokens & {"test", "tests", "sealed"})


def _jsonl_paths(roots: Iterable[Path]) -> list[Path]:
    result: set[Path] = set()
    for root in roots:
        if root.is_file() and root.suffix.lower() == ".jsonl":
            if not _excluded(root):
                result.add(root.resolve())
            continue
        if not root.exists():
            raise FileNotFoundError(root)
        def onerror(error: OSError) -> None:
            raise error

        for current, directories, filenames in os.walk(root, topdown=True, onerror=onerror):
            current_path = Path(current)
            directories[:] = [
                name for name in directories if name.lower() not in EXCLUDED_PARTS
            ]
            for filename in filenames:
                if filename.lower().endswith(".jsonl"):
                    path = current_path / filename
                    if not _excluded(path):
                        result.add(path.resolve())
    return sorted(result)


def _json_metadata_paths(roots: Iterable[Path]) -> list[Path]:
    result: set[Path] = set()

    def onerror(error: OSError) -> None:
        raise error

    for root in roots:
        if root.is_file() and root.suffix.lower() == ".json":
            if not _excluded(root):
                result.add(root.resolve())
            continue
        if not root.exists():
            raise FileNotFoundError(root)
        for current, directories, filenames in os.walk(root, topdown=True, onerror=onerror):
            current_path = Path(current)
            directories[:] = [
                name
                for name in directories
                if name.lower() not in EXCLUDED_PARTS | {"raw", "raw-questions"}
            ]
            relative_parts = {part.lower() for part in current_path.parts}
            relevant_tree = "clevrer" in str(current_path).lower() or {
                "docs", "pretrained_reuse"
            }.issubset(relative_parts) or {"manifests", "pretrained_reuse"}.issubset(
                relative_parts
            )
            if not relevant_tree:
                continue
            for filename in filenames:
                if filename.lower().endswith(".json") and not _excluded(
                    current_path / filename
                ) and filename.lower() not in {
                    "train-questions.json", "validation-questions.json", "receipt.json"
                }:
                    result.add((current_path / filename).resolve())
    return sorted(result)


def _read_exposures(paths: Iterable[Path]) -> tuple[set[int], int]:
    exposed: set[int] = set()
    rows_count = 0
    for path in paths:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"JSONL row must be an object: {path}:{line_number}")
                rows_count += 1
                exposed.update(exposed_scene_ids([row]))
    return exposed, rows_count


def _json_exposures(paths: Iterable[Path]) -> tuple[set[int], int]:
    exposed: set[int] = set()
    object_count = 0

    def visit(value: Any, source_context: bool, manifest_context: bool) -> None:
        nonlocal object_count
        if isinstance(value, dict):
            object_count += 1
            local_context = source_context or any(
                key.lower() in SOURCE_KEYS
                and isinstance(child, str)
                and "clevrer" in child.lower()
                for key, child in value.items()
            )
            row = dict(value)
            if local_context and not any(key.lower() in SOURCE_KEYS for key in row):
                row["source"] = "CLEVRER"
            exposed.update(exposed_scene_ids([row]))
            for child in value.values():
                visit(child, local_context, manifest_context)
        elif isinstance(value, list):
            if manifest_context and all(isinstance(item, int) for item in value):
                exposed.update(item for item in value if 0 <= item < 10_000)
                object_count += len(value)
            else:
                for child in value:
                    visit(child, source_context, manifest_context)

    for path in paths:
        text = path.read_text(encoding="utf-8")
        manifest_context = "clevrer" in path.name.lower() and path.name.lower().endswith(
            "excluded-scenes-20261011.json"
        )
        lowered = text.lower()
        identity_markers = (
            '"scene_index"', '"scene_group_id"', '"video_filename"',
            '"dataset_id"', 'mit-ibm/clevrer', 'clevrer-train-', 'clevrer:train:',
            'clevrer-validation-', 'clevrer:validation:',
        )
        if not manifest_context and not any(marker in lowered for marker in identity_markers):
            continue
        try:
            value = json.loads(text)
            documents = [value]
        except json.JSONDecodeError as document_error:
            documents = []
            try:
                for _line_number, line in enumerate(text.splitlines(), 1):
                    if line.strip():
                        documents.append(json.loads(line))
            except json.JSONDecodeError as line_error:
                raise ValueError(f"invalid JSON metadata in {path}") from line_error
            if not documents:
                raise ValueError(f"empty JSON metadata file: {path}") from document_error
        for value in documents:
            visit(value, False, manifest_context)
    return exposed, object_count


def _cached_scene_ids(roots: Iterable[Path]) -> set[int]:
    result: set[int] = set()
    pattern = re.compile(r"^video_(\d{5})\.mp4$", re.IGNORECASE)

    def onerror(error: OSError) -> None:
        raise error

    for root in roots:
        if not root.exists():
            continue
        for _current, directories, filenames in os.walk(root, topdown=True, onerror=onerror):
            directories[:] = [
                name for name in directories if name.lower() not in EXCLUDED_PARTS
            ]
            for filename in filenames:
                match = pattern.match(filename)
                if match:
                    result.add(int(match.group(1)))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-questions", type=Path, required=True)
    parser.add_argument("--train-questions-sha256", required=True)
    parser.add_argument("--scan-root", type=Path, action="append", required=True)
    parser.add_argument("--media-root", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    official = list(
        iter_train_scene_ids(
            args.train_questions, expected_sha256=args.train_questions_sha256
        )
    )
    jsonl_paths = _jsonl_paths(args.scan_root)
    exposed, row_count = _read_exposures(jsonl_paths)
    json_paths = _json_metadata_paths(args.scan_root)
    json_exposed, json_object_count = _json_exposures(json_paths)
    exposed.update(json_exposed)
    cached = _cached_scene_ids(args.media_root)
    candidates = candidate_scene_ids(official, exposed=exposed, cached_media=cached)
    metadata_path_payload = "\n".join(
        str(path) for path in sorted([*jsonl_paths, *json_paths])
    ).encode("utf-8")
    receipt = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "method": "metadata and filenames only; media bytes were not read",
        "audit_script_sha256": sha256_file(Path(__file__)),
        "scan_root_count": len(args.scan_root),
        "media_root_count": len(args.media_root),
        "metadata_paths_sha256": hashlib.sha256(metadata_path_payload).hexdigest(),
        "train_questions_path": str(args.train_questions.resolve()),
        "train_questions_sha256": sha256_file(args.train_questions),
        "official_scene_ids_sha256": sha256_ids(official),
        "official_scene_count": len(official),
        "jsonl_file_count": len(jsonl_paths),
        "jsonl_row_count": row_count,
        "json_file_count": len(json_paths),
        "json_object_count": json_object_count,
        "metadata_exposed_scene_ids_sha256": sha256_ids(exposed),
        "metadata_exposed_scene_count": len(exposed),
        "metadata_exposed_scene_ids": sorted(exposed),
        "cached_media_scene_ids_sha256": sha256_ids(cached),
        "cached_media_scene_count": len(cached),
        "cached_media_scene_ids": sorted(cached),
        "candidate_scene_count": len(candidates),
        "candidate_scene_ids_sha256": sha256_ids(candidates),
        "candidate_scene_ids": candidates,
        "unexamined_for_content_or_media_overlap": True,
    }
    write_receipt(args.output, receipt)
    print(json.dumps({k: v for k, v in receipt.items() if k != "candidate_scene_ids"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
