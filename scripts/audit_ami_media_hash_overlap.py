"""Verify AMI media receipt hashes and report exact train/validation duplicates."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _entries(receipt: Any) -> list[dict[str, Any]]:
    if isinstance(receipt, list):
        return receipt
    if isinstance(receipt, dict) and isinstance(receipt.get("files"), list):
        return receipt["files"]
    raise ValueError("receipt must be a list or contain a files list")


def _field(entry: dict[str, Any], *names: str) -> str:
    for name in names:
        value = entry.get(name)
        if value is not None and str(value):
            return str(value)
    raise ValueError(f"missing receipt field; expected one of {names}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_receipts(
    receipt_paths: list[Path],
    media_root: Path,
    train_meetings: set[str],
    validation_meetings: set[str],
) -> dict[str, Any]:
    overlap = train_meetings & validation_meetings
    if overlap:
        raise ValueError(f"meeting sets overlap: {sorted(overlap)}")

    records: dict[tuple[str, str], dict[str, Any]] = {}
    receipt_hashes: list[dict[str, str]] = []
    for receipt_path in receipt_paths:
        receipt_bytes = receipt_path.read_bytes()
        receipt = json.loads(receipt_bytes)
        receipt_hashes.append(
            {
                "name": receipt_path.name,
                "sha256": hashlib.sha256(receipt_bytes).hexdigest(),
            }
        )
        for entry in _entries(receipt):
            meeting = _field(entry, "meeting")
            filename = _field(entry, "file", "filename")
            if meeting in train_meetings:
                split = "train"
            elif meeting in validation_meetings:
                split = "validation"
            else:
                raise ValueError(f"meeting absent from split map: {meeting}")

            kind = entry.get("kind") or entry.get("type")
            if not kind:
                kind = "audio" if filename.lower().endswith(".wav") else "video"
            if kind not in {"audio", "video"}:
                raise ValueError(f"unsupported media kind: {kind}")

            path = media_root / meeting / kind / filename
            expected_size = int(entry["bytes"])
            expected_hash = _field(entry, "sha256").lower()
            if not path.is_file():
                raise FileNotFoundError(path)
            actual_size = path.stat().st_size
            actual_hash = _sha256(path)
            if actual_size != expected_size:
                raise ValueError(f"size mismatch for {path}")
            if actual_hash != expected_hash:
                raise ValueError(f"SHA-256 mismatch for {path}")

            key = (meeting, filename)
            record = {
                "split": split,
                "meeting": meeting,
                "kind": kind,
                "file": filename,
                "bytes": actual_size,
                "sha256": actual_hash,
                "receipt_refs": [receipt_path.name],
            }
            previous = records.get(key)
            if previous is None:
                records[key] = record
            else:
                comparable = ("split", "kind", "bytes", "sha256")
                if any(previous[field] != record[field] for field in comparable):
                    raise ValueError(f"conflicting receipts for {meeting}/{filename}")
                previous["receipt_refs"].append(receipt_path.name)

    by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    ordered_records = sorted(
        records.values(),
        key=lambda row: (row["split"], row["meeting"], row["file"]),
    )
    for record in ordered_records:
        by_hash[record["sha256"]].append(record)
    cross_split_duplicates = []
    for digest, matching in sorted(by_hash.items()):
        splits = {row["split"] for row in matching}
        if len(splits) > 1:
            cross_split_duplicates.append(
                {
                    "sha256": digest,
                    "files": [
                        {"split": row["split"], "meeting": row["meeting"], "file": row["file"]}
                        for row in matching
                    ],
                }
            )

    return {
        "schema": "ami-media-hash-overlap-audit-v1",
        "receipts": receipt_hashes,
        "record_count": len(ordered_records),
        "train_record_count": sum(row["split"] == "train" for row in ordered_records),
        "validation_record_count": sum(row["split"] == "validation" for row in ordered_records),
        "cross_split_exact_duplicate_group_count": len(cross_split_duplicates),
        "cross_split_exact_duplicates": cross_split_duplicates,
        "files": ordered_records,
        "scope_note": "Exact source-file SHA-256 only; not a perceptual or near-duplicate audit.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipts", nargs="+", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--train-meetings", nargs="+", required=True)
    parser.add_argument("--validation-meetings", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite output: {args.output}")
    result = audit_receipts(
        args.receipts,
        args.media_root,
        set(args.train_meetings),
        set(args.validation_meetings),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
