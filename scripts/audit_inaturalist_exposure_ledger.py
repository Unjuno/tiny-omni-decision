"""Reconcile a frozen iNaturalist exposure ledger with an external cohort.

This is a read-only audit. It emits aggregate counts and file hashes only; raw
observation IDs, photo IDs, and observer identifiers are never written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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


def audit_local_inventory(ledger: dict[str, Any], inventory_root: Path) -> dict[str, Any]:
    """Scan first-level iNaturalist artifact directories without following links."""
    source_id = str(ledger.get("source_id", ""))
    prior_observer_digests = {
        digest
        for item in ledger.get("usage", [])
        for digest in item.get("identity_hashes_by_kind", {}).get(
            "observer_group_sha256", []
        )
    }
    raw_observer_groups: set[str] = set()
    file_hashes: list[dict[str, str]] = []
    row_count = 0
    skipped_unreadable_files = 0
    directories_scanned = 0

    for artifact_dir in sorted(inventory_root.iterdir(), key=lambda item: item.name):
        try:
            if (
                not artifact_dir.name.casefold().startswith("inat-")
                or artifact_dir.is_symlink()
                or not artifact_dir.is_dir()
            ):
                continue
        except OSError:
            continue
        directories_scanned += 1
        pending = [(artifact_dir, 0)]
        while pending:
            current_dir, depth = pending.pop()
            try:
                entries = list(os.scandir(current_dir))
            except OSError:
                skipped_unreadable_files += 1
                continue
            for entry in entries:
                try:
                    if entry.is_symlink():
                        continue
                    if entry.is_dir(follow_symlinks=False) and depth < 2:
                        pending.append((Path(entry.path), depth + 1))
                        continue
                    if not entry.is_file(follow_symlinks=False) or not entry.name.endswith(
                        ".jsonl"
                    ):
                        continue
                    path = Path(entry.path)
                    payload = path.read_bytes()
                    digest = hashlib.sha256(payload).hexdigest()
                    file_hashes.append(
                        {"path": path.relative_to(inventory_root).as_posix(), "sha256": digest}
                    )
                    for line_number, line in enumerate(payload.splitlines(), 1):
                        if not line.strip():
                            continue
                        row = json.loads(line)
                        if not isinstance(row, dict):
                            raise ValueError(f"{path}:{line_number}: expected a JSON object")
                        group = row.get("observer_group_sha256")
                        if group is not None:
                            raw_observer_groups.add(str(group))
                            row_count += 1
                except (OSError, json.JSONDecodeError):
                    skipped_unreadable_files += 1

    ledger_matches = {
        group
        for group in raw_observer_groups
        if identity_digest(source_id, "observer_group_sha256", group)
        in prior_observer_digests
    }
    file_hashes.sort(key=lambda item: item["path"])
    canonical_inventory = json.dumps(
        file_hashes, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "schema": "tiny-omni-decision/inaturalist-local-inventory-v1",
        "root": str(inventory_root),
        "directories_scanned": directories_scanned,
        "jsonl_files_scanned": len(file_hashes),
        "jsonl_file_manifest_sha256": hashlib.sha256(canonical_inventory).hexdigest(),
        "rows_with_observer_group_hash": row_count,
        "unique_observer_groups": len(raw_observer_groups),
        "unique_groups_matching_ledger": len(ledger_matches),
        "unique_groups_missing_from_ledger": len(raw_observer_groups - ledger_matches),
        "unreadable_files_or_directories": skipped_unreadable_files,
        "raw_ids_written": False,
        "ledger_history_status": ledger.get("history_status", "UNKNOWN"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--cohort-root", type=Path, required=True)
    parser.add_argument("--inventory-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = audit_paths(args.ledger, args.cohort_root)
    if args.inventory_root:
        ledger = json.loads(args.ledger.read_text(encoding="utf-8"))
        result["local_inventory"] = audit_local_inventory(ledger, args.inventory_root)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)
    return 0 if result["history_reconciled"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
