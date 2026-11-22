"""Fail closed when a proposed source cohort has prior or unknown use."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def group_digest(source_id: str, revision: str, group_id: str) -> str:
    """Hash a source-scoped group identity without persisting the raw identifier."""
    if not source_id.strip() or not revision.strip() or not group_id.strip():
        raise ValueError("source ID, revision, and group ID must be nonempty")
    return hashlib.sha256(
        _canonical([source_id.strip(), revision.strip(), group_id.strip()]).encode("utf-8")
    ).hexdigest()


def read_group_ids(path: Path, field: str) -> list[str]:
    """Read a JSON string array or JSONL records and return unique group IDs."""
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        values = [json.loads(line) for line in raw.splitlines() if line.strip()]
        ids = list(dict.fromkeys(str(value[field]) for value in values))
    else:
        value = json.loads(raw)
        if not isinstance(value, list):
            raise ValueError("group input JSON must be a list")
        ids = [str(item) for item in value]
    if not ids or any(not item.strip() for item in ids):
        raise ValueError("group input must contain nonempty IDs")
    if len(ids) != len(set(ids)):
        raise ValueError("group input contains duplicate IDs")
    return ids


def check_cohort(
    ledger: dict,
    *,
    source_id: str,
    revision: str,
    group_ids: list[str],
) -> dict:
    """Return a fail-closed cohort decision and exact known-overlap hashes."""
    if not group_ids or len(group_ids) != len(set(group_ids)):
        raise ValueError("proposed cohort must contain unique group IDs")

    sources = ledger.get("sources")
    if not isinstance(sources, list):
        raise ValueError("ledger sources must be a list")
    matches = [
        item
        for item in sources
        if item.get("source_id") == source_id and item.get("revision") == revision
    ]
    if len(matches) != 1:
        return {
            "decision": "BLOCKED_UNKNOWN_SOURCE_OR_REVISION",
            "source_id": source_id,
            "revision": revision,
            "proposed_group_count": len(group_ids),
            "overlap_count": None,
        }

    source = matches[0]
    known = {
        digest
        for entry in source.get("usage", [])
        for digest in entry.get("group_identity_sha256", [])
    }
    proposed = {group_digest(source_id, revision, group_id) for group_id in group_ids}
    overlap = sorted(known & proposed)
    if overlap:
        decision = "BLOCKED_PRIOR_GROUP_USE"
    elif source.get("history_status") != "COMPLETE":
        decision = "BLOCKED_INCOMPLETE_HISTORY"
    else:
        decision = "CLEAR_WITHIN_DECLARED_HISTORY"
    return {
        "decision": decision,
        "source_id": source_id,
        "revision": revision,
        "proposed_group_count": len(group_ids),
        "overlap_count": len(overlap),
        "overlap_group_identity_sha256": overlap,
        "history_status": source.get("history_status", "UNKNOWN"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--groups", required=True, type=Path)
    parser.add_argument("--jsonl-group-field", default="group_id")
    args = parser.parse_args()

    ledger = json.loads(args.ledger.read_text(encoding="utf-8"))
    group_ids = read_group_ids(args.groups, args.jsonl_group_field)
    result = check_cohort(
        ledger,
        source_id=args.source_id,
        revision=args.revision,
        group_ids=group_ids,
    )
    overlap_hashes = result.pop("overlap_group_identity_sha256", [])
    result["overlap_group_identity_sha256_sample"] = overlap_hashes[:10]
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0 if result["decision"] == "CLEAR_WITHIN_DECLARED_HISTORY" else 2


if __name__ == "__main__":
    sys.exit(main())
