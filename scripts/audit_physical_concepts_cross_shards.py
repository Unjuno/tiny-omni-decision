"""Check exact content identities across pinned Physical Concepts shard audits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

IDENTITY_FIELDS = {
    "record_sha256": "serialized_records",
    "rgb_sequence_sha256": "rgb_sequences",
    "mask_sequence_sha256": "mask_sequences",
}


def _load_report(path: Path, split: str) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "complete":
        raise ValueError(f"{split} shard audit is not complete")
    if not report.get("records") or report["records"] != report.get("unique_records"):
        raise ValueError(f"{split} shard contains duplicate or zero records")
    for field in IDENTITY_FIELDS:
        values = report.get(field)
        if not isinstance(values, list) or len(values) != report["records"]:
            raise ValueError(f"{split} shard has invalid identity list: {field}")
        if len(set(values)) != len(values):
            raise ValueError(f"{split} shard contains duplicate identities: {field}")
    return report


def audit_cross_shards(train_path: Path, validation_path: Path) -> dict[str, Any]:
    """Fail closed on any exact row, RGB-sequence, or mask-sequence overlap."""
    train = _load_report(train_path, "train")
    validation = _load_report(validation_path, "validation")
    if train["object_name"] == validation["object_name"]:
        raise ValueError("train and validation must use different source shards")
    overlap = {
        label: len(set(train[field]) & set(validation[field]))
        for field, label in IDENTITY_FIELDS.items()
    }
    if any(overlap.values()):
        raise ValueError(f"cross-shard exact content overlap: {overlap}")
    return {
        "status": "complete_exact_cross_shard_identity_audit",
        "assignment": {
            "train": train["object_name"],
            "validation": validation["object_name"],
        },
        "shards": [
            {
                "object_name": report["object_name"],
                "generation": report["generation"],
                "records": report["records"],
                "source_sha256": report["source_sha256"],
                "unique_records": report["unique_records"],
                "unique_rgb_sequences": report["unique_rgb_sequences"],
                "unique_mask_sequences": report["unique_mask_sequences"],
            }
            for report in (train, validation)
        ],
        "combined_records": train["records"] + validation["records"],
        "cross_shard_exact_overlaps": overlap,
        "limitations": [
            "Only the listed shards were audited.",
            "No scene or procedural-generation identity is exposed by these records.",
            "Exact hashes do not rule out near-duplicate scenes or related generated"
            " configurations.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-report", type=Path, required=True)
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_cross_shards(args.train_report, args.validation_report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.write_text(serialized, encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"receipt_sha256={hashlib.sha256(serialized.encode('utf-8')).hexdigest()}")


if __name__ == "__main__":
    main()
