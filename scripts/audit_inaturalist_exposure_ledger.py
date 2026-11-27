"""Reconcile a frozen iNaturalist exposure ledger with an external cohort.

This is a read-only audit. It emits aggregate counts and file hashes only; raw
observation IDs, photo IDs, and observer identifiers are never written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


def identity_digest(source_id: str, identity_kind: str, identity_id: Any) -> str:
    """Match the canonical array-based identity hashing scheme in the ledger."""
    payload = json.dumps(
        [source_id, identity_kind, identity_id],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            for key in ("observation_id", "photo_id", "observer_group_sha256"):
                if key not in value:
                    raise ValueError(f"{path}:{line_number}: missing {key}")
            rows.append(value)
    return rows


def _row_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["observation_id"]), str(row["photo_id"])


def audit_cohorts(
    ledger: dict[str, Any],
    source_train: list[dict[str, Any]],
    source_validation: list[dict[str, Any]],
    retained_train: list[dict[str, Any]],
    retained_validation: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return a privacy-minimized reconciliation summary for one frozen cohort."""
    source_id = str(ledger.get("source_id", ""))
    if not source_id:
        raise ValueError("ledger is missing source_id")

    usage = ledger.get("usage")
    if not isinstance(usage, list):
        raise ValueError("ledger usage must be a list")
    prior_observer_digests = {
        digest
        for item in usage
        for digest in item.get("identity_hashes_by_kind", {}).get(
            "observer_group_sha256", []
        )
    }

    source_rows = source_train + source_validation
    retained_rows = retained_train + retained_validation
    source_keys = [_row_key(row) for row in source_rows]
    retained_keys = [_row_key(row) for row in retained_rows]
    if len(set(source_keys)) != len(source_keys):
        raise ValueError("source cohort contains duplicate observation/photo identities")
    if len(set(retained_keys)) != len(retained_keys):
        raise ValueError("retained cohort contains duplicate observation/photo identities")
    if not set(retained_keys).issubset(source_keys):
        raise ValueError("retained rows are not a subset of the frozen source cohort")

    train_groups = {str(row["observer_group_sha256"]) for row in retained_train}
    validation_groups = {
        str(row["observer_group_sha256"]) for row in retained_validation
    }
    if train_groups & validation_groups:
        raise ValueError("retained observer groups cross train/validation")

    retained_set = set(retained_keys)
    excluded_rows = [row for row in source_rows if _row_key(row) not in retained_set]
    excluded_without_ledger_match = [
        row
        for row in excluded_rows
        if identity_digest(
            source_id, "observer_group_sha256", str(row["observer_group_sha256"])
        )
        not in prior_observer_digests
    ]
    source_observer_matches = sum(
        identity_digest(
            source_id, "observer_group_sha256", str(row["observer_group_sha256"])
        )
        in prior_observer_digests
        for row in source_rows
    )
    retained_prior_matches = sum(
        identity_digest(
            source_id, "observer_group_sha256", str(row["observer_group_sha256"])
        )
        in prior_observer_digests
        for row in retained_rows
    )

    metadata_previous_count = int(
        ledger.get("comparison_metadata", {}).get("previous_observer_count", 0)
    )
    return {
        "schema": "tiny-omni-decision/inaturalist-ledger-reconciliation-v1",
        "source_id": source_id,
        "ledger_history_status": ledger.get("history_status", "UNKNOWN"),
        "ledger_unique_observer_digests": len(prior_observer_digests),
        "preflight_previous_observer_count": metadata_previous_count,
        "preflight_minus_ledger_observer_count": (
            metadata_previous_count - len(prior_observer_digests)
        ),
        "source_rows": len(source_rows),
        "source_train_rows": len(source_train),
        "source_validation_rows": len(source_validation),
        "retained_rows": len(retained_rows),
        "retained_train_rows": len(retained_train),
        "retained_validation_rows": len(retained_validation),
        "excluded_rows": len(excluded_rows),
        "source_rows_matching_ledger_observer": source_observer_matches,
        "retained_rows_matching_ledger_observer": retained_prior_matches,
        "excluded_rows_without_ledger_observer_match": len(
            excluded_without_ledger_match
        ),
        "unique_excluded_observer_groups_without_ledger_match": len(
            {str(row["observer_group_sha256"]) for row in excluded_without_ledger_match}
        ),
        "retained_classes": sorted({str(row.get("label", "UNKNOWN")) for row in retained_rows}),
        "retained_train_validation_observer_overlap": len(
            train_groups & validation_groups
        ),
        "history_reconciled": (
            ledger.get("history_status") == "COMPLETE"
            and metadata_previous_count == len(prior_observer_digests)
            and not excluded_without_ledger_match
            and retained_prior_matches == 0
        ),
    }


def audit_paths(ledger_path: Path, cohort_root: Path) -> dict[str, Any]:
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    base = cohort_root
    source_train_path = base / "source" / "train.jsonl"
    source_validation_path = base / "source" / "validation.jsonl"
    retained_train_path = base / "filtered" / "train.jsonl"
    retained_validation_path = base / "filtered" / "validation.jsonl"
    metadata_path = base / "filtered" / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    summary = audit_cohorts(
        {**ledger, "comparison_metadata": metadata.get("summary", {})},
        read_jsonl(source_train_path),
        read_jsonl(source_validation_path),
        read_jsonl(retained_train_path),
        read_jsonl(retained_validation_path),
    )
    summary["input_sha256"] = {
        "ledger": sha256_file(ledger_path),
        "source_train": sha256_file(source_train_path),
        "source_validation": sha256_file(source_validation_path),
        "retained_train": sha256_file(retained_train_path),
        "retained_validation": sha256_file(retained_validation_path),
        "filtered_metadata": sha256_file(metadata_path),
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--cohort-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = audit_paths(args.ledger, args.cohort_root)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)
    return 0 if result["history_reconciled"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
