"""Audit canonical scene-configuration overlap across the frozen internal split."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
import zipfile
from pathlib import Path
from typing import Any, TextIO

from scripts.audit_clevr4_content_overlap import _load_jsonl, sha256_file


class _StreamingJSON:
    def __init__(self, stream: TextIO) -> None:
        self.stream = stream
        self.decoder = json.JSONDecoder()
        self.buffer = stream.read(65536)
        self.position = 0

    def _refill(self) -> bool:
        chunk = self.stream.read(65536)
        if not chunk:
            return False
        self.buffer = self.buffer[self.position :] + chunk
        self.position = 0
        return True

    def _skip_ws(self) -> None:
        while True:
            while self.position < len(self.buffer) and self.buffer[self.position].isspace():
                self.position += 1
            if self.position < len(self.buffer) or not self._refill():
                return

    def _decode(self) -> Any:
        while True:
            try:
                value, position = self.decoder.raw_decode(self.buffer, self.position)
                self.position = position
                return value
            except json.JSONDecodeError:
                if not self._refill():
                    raise

    def _skip_value(self) -> None:
        self._skip_ws()
        if self.position >= len(self.buffer) and not self._refill():
            raise ValueError("unexpected EOF while skipping a scene record")
        first = self.buffer[self.position]
        if first not in "{[":
            self._decode()
            return
        closing = {"{": "}", "[": "]"}
        stack = [closing[first]]
        self.position += 1
        in_string = False
        escaped = False
        while stack:
            if self.position >= len(self.buffer) and not self._refill():
                raise ValueError("unexpected EOF inside a skipped scene record")
            char = self.buffer[self.position]
            self.position += 1
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char in "{[":
                stack.append(closing[char])
            elif char in "}]":
                if not stack or stack.pop() != char:
                    raise ValueError("mismatched bracket in skipped JSON value")

    def iter_object(self, wanted_ids: set[str]):
        self._skip_ws()
        if self.buffer[self.position] != "{":
            raise ValueError("scene annotation JSON must be a top-level object")
        self.position += 1
        while True:
            self._skip_ws()
            if self.position >= len(self.buffer) and not self._refill():
                raise ValueError("unexpected EOF in top-level scene object")
            if self.buffer[self.position] == "}":
                return
            if self.buffer[self.position] == ",":
                self.position += 1
                self._skip_ws()
            key = self._decode()
            if not isinstance(key, str):
                raise ValueError("scene annotation key must be a string")
            self._skip_ws()
            if self.buffer[self.position] != ":":
                if self.position >= len(self.buffer) and self._refill():
                    self._skip_ws()
                if self.buffer[self.position] != ":":
                    raise ValueError("expected colon after scene annotation key")
            self.position += 1
            self._skip_ws()
            if key in wanted_ids:
                value = self._decode()
                yield key, value
            else:
                self._skip_value()
            self.buffer = self.buffer[self.position :]
            self.position = 0


def _round_numbers(value: Any, precision: int) -> Any:
    if isinstance(value, float):
        rounded = round(value, precision)
        if not math.isfinite(rounded):
            raise ValueError("non-finite coordinate in scene metadata")
        return rounded
    if isinstance(value, list):
        return [_round_numbers(item, precision) for item in value]
    if isinstance(value, dict):
        return {key: _round_numbers(item, precision) for key, item in value.items()}
    return value


def scene_fingerprint(scene: dict[str, Any], *, precision: int | None = None) -> str:
    canonical = {
        key: scene[key] for key in ("directions", "objects", "relationships") if key in scene
    }
    if "objects" not in canonical:
        raise ValueError("scene metadata has no objects list")
    if precision is not None:
        canonical = _round_numbers(canonical, precision)
    serialized = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _cross_split_collisions(
    train: dict[str, str], validation: dict[str, str]
) -> list[dict[str, Any]]:
    train_by_hash: dict[str, list[str]] = {}
    val_by_hash: dict[str, list[str]] = {}
    for sample_id, digest in train.items():
        train_by_hash.setdefault(digest, []).append(sample_id)
    for sample_id, digest in validation.items():
        val_by_hash.setdefault(digest, []).append(sample_id)
    return [
        {"train_ids": train_by_hash[digest], "validation_ids": val_by_hash[digest]}
        for digest in sorted(train_by_hash.keys() & val_by_hash.keys())
    ]


def run_audit(
    *,
    archive: Path,
    train_labels: Path,
    validation_labels: Path,
    train_media_manifest: Path,
    validation_media_manifest: Path,
    output: Path,
    member: str = "whole_scene_data.json",
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    train_rows = _load_jsonl(train_labels)
    validation_rows = _load_jsonl(validation_labels)
    train_media_rows = _load_jsonl(train_media_manifest)
    validation_media_rows = _load_jsonl(validation_media_manifest)
    if any(row.get("source_split") != "train" for row in train_rows + validation_rows):
        raise ValueError("internal train/validation manifests must contain source TRAIN only")
    train_ids = [row["id"] for row in train_rows]
    validation_ids = [row["id"] for row in validation_rows]
    if train_ids != [row["id"] for row in train_media_rows]:
        raise ValueError("train label IDs/order differ from media manifest")
    if validation_ids != [row["id"] for row in validation_media_rows]:
        raise ValueError("validation label IDs/order differ from media manifest")
    train_id_set = set(train_ids)
    wanted_ids = set(train_ids) | set(validation_ids)
    if len(wanted_ids) != len(train_ids) + len(validation_ids):
        raise ValueError("duplicate or overlapping IDs in frozen internal split manifests")
    ranked_ids = sorted(
        wanted_ids,
        key=lambda sample_id: (
            hashlib.sha256(f"17:{sample_id}".encode()).digest(),
            sample_id,
        ),
    )
    expected_validation_ids = set(ranked_ids[: math.ceil(len(wanted_ids) * 0.1)])
    if expected_validation_ids != set(validation_ids):
        raise ValueError("validation IDs do not reproduce the frozen seed-17 split")

    started = time.perf_counter()
    exact_train: dict[str, str] = {}
    rounded_train: dict[str, str] = {}
    exact_validation: dict[str, str] = {}
    rounded_validation: dict[str, str] = {}
    found: set[str] = set()
    with zipfile.ZipFile(archive) as source:
        info = source.getinfo(member)
        with source.open(member) as raw:
            import io

            text = io.TextIOWrapper(raw, encoding="utf-8")
            for index, (sample_id, scene) in enumerate(
                _StreamingJSON(text).iter_object(wanted_ids), start=1
            ):
                if sample_id in wanted_ids:
                    found.add(sample_id)
                    exact = scene_fingerprint(scene)
                    rounded = scene_fingerprint(scene, precision=3)
                    if sample_id in train_id_set:
                        exact_train[sample_id] = exact
                        rounded_train[sample_id] = rounded
                    else:
                        exact_validation[sample_id] = exact
                        rounded_validation[sample_id] = rounded
                if index % 20000 == 0:
                    print(f"streamed scene rows: {index}", flush=True)

    missing = sorted(wanted_ids - found)
    if missing:
        raise ValueError(f"scene metadata missing {len(missing)} frozen source-TRAIN IDs")
    exact_collisions = _cross_split_collisions(exact_train, exact_validation)
    rounded_collisions = _cross_split_collisions(rounded_train, rounded_validation)
    result = {
        "status": "COMPLETE_SCENE_CONFIGURATION_OVERLAP_AUDIT",
        "archive": str(archive),
        "archive_sha512_recomputed": False,
        "annotation_member": member,
        "annotation_member_bytes": info.file_size,
        "annotation_member_crc32": f"{info.CRC:08x}",
        "source_split_scope": (
            "only IDs in frozen internal train/validation manifests, both from official TRAIN"
        ),
        "official_validation_scene_records_used_for_hashing": False,
        "train_count": len(train_ids),
        "validation_count": len(validation_ids),
        "train_id_order_sha256": hashlib.sha256(("\n".join(train_ids) + "\n").encode()).hexdigest(),
        "validation_id_order_sha256": hashlib.sha256(
            ("\n".join(validation_ids) + "\n").encode()
        ).hexdigest(),
        "train_labels_sha256": sha256_file(train_labels),
        "validation_labels_sha256": sha256_file(validation_labels),
        "train_media_manifest_sha256": sha256_file(train_media_manifest),
        "validation_media_manifest_sha256": sha256_file(validation_media_manifest),
        "split_reproduction": (
            "PASS seed=17; SHA256(seed:id) rank; first ceil(0.1*N) validation IDs"
        ),
        "matched_scene_records": len(found),
        "exact_scene_config_cross_split_group_count": len(exact_collisions),
        "exact_scene_config_cross_split_asset_count": sum(
            len(row["train_ids"]) + len(row["validation_ids"]) for row in exact_collisions
        ),
        "rounded_3dp_scene_config_cross_split_group_count": len(rounded_collisions),
        "rounded_3dp_scene_config_cross_split_asset_count": sum(
            len(row["train_ids"]) + len(row["validation_ids"]) for row in rounded_collisions
        ),
        "exact_collisions": exact_collisions,
        "rounded_collisions": rounded_collisions,
        "elapsed_seconds": time.perf_counter() - started,
        "source_files_modified": False,
        "media_files_written": False,
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--train-labels", type=Path, required=True)
    parser.add_argument("--validation-labels", type=Path, required=True)
    parser.add_argument("--train-media-manifest", type=Path, required=True)
    parser.add_argument("--validation-media-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--member", default="whole_scene_data.json")
    args = parser.parse_args()
    print(
        json.dumps(
            run_audit(
                archive=args.archive,
                train_labels=args.train_labels,
                validation_labels=args.validation_labels,
                train_media_manifest=args.train_media_manifest,
                validation_media_manifest=args.validation_media_manifest,
                output=args.output,
                member=args.member,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
