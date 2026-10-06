from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tiny_omni_decision.teacher_cache import (  # noqa: E402
    canonical_cache_bytes,
    sample_id_order_sha256,
)

PROTECTED = re.compile(r"(^|[-_.])(audit|sealed|heldout)([-_.]|$)", re.I)


def _protected(path: Path) -> None:
    candidates = (path, path.resolve(strict=False))
    if any(PROTECTED.search(part) for candidate in candidates for part in candidate.parts):
        raise ValueError(f"protected audit/sealed/heldout path: {path}")


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _nonfinite(value: str) -> None:
    raise ValueError(f"nonfinite JSON value: {value}")


def _read_rows(path: Path) -> tuple[list[dict[str, object]], str]:
    _protected(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError("input JSONL must be a regular non-symlink file")
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeError as exc:
        raise ValueError("input JSONL must be UTF-8") from exc
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(
                line,
                object_pairs_hook=_object,
                parse_constant=_nonfinite,
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"line {line_number}: invalid strict JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"line {line_number}: expected JSON object")
        rows.append(value)
    return rows, hashlib.sha256(raw).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and canonicalize already-enriched Q0 option-logit cache rows; "
            "this command does not run a model"
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    temp = args.output.with_name(args.output.name + ".tmp")
    try:
        _protected(args.input)
        _protected(args.output)
        _protected(temp)
        if args.output.exists() or args.output.is_symlink():
            raise ValueError("output already exists; teacher caches are immutable")
        if temp.exists() or temp.is_symlink():
            raise ValueError("temporary output already exists; inspect before retrying")
        rows, source_sha256 = _read_rows(args.input)
        payload, digest, count = canonical_cache_bytes(rows)
        order_digest = sample_id_order_sha256(rows)
        temp.mkdir(parents=False)
        try:
            (temp / "teacher-cache.jsonl").write_bytes(payload)
            manifest = {
                "schema_version": 1,
                "kind": "postquant_teacher_option_cache",
                "row_count": count,
                "cache_sha256": digest,
                "sample_id_order_sha256": order_digest,
                "source_jsonl_sha256": source_sha256,
            }
            (temp / "manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(temp, args.output)
        except BaseException:
            shutil.rmtree(temp, ignore_errors=True)
            raise
        print(
            json.dumps(
                {
                    "status": "CREATED",
                    "row_count": count,
                    "cache_sha256": digest,
                    "sample_id_order_sha256": order_digest,
                }
            )
        )
        return 0
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        shutil.rmtree(temp, ignore_errors=True)
        print(json.dumps({"status": "BLOCKED", "error": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
