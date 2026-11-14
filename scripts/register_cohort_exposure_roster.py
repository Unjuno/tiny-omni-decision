"""Register a metadata-only group exposure roster without clearing history."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
LEDGER_FORMAT = "pretrained_reuse_cohort_usage_ledger_v1"


def register_roster(
    ledger: dict[str, Any],
    roster: dict[str, Any],
    *,
    roster_path: str,
    roster_sha256: str,
) -> dict[str, Any]:
    if ledger.get("format") != LEDGER_FORMAT:
        raise ValueError("unsupported cohort ledger format")
    if roster.get("schema_version") != 1:
        raise ValueError("unsupported exposure roster schema")
    if roster.get("exposure") != "metadata_only_utterance_ids_read_no_media_or_content":
        raise ValueError("roster does not declare metadata-only exposure")
    if not HEX_SHA256.fullmatch(roster_sha256.lower()):
        raise ValueError("roster SHA-256 is malformed")

    group_count = roster.get("unique_parent_clip_count")
    if not isinstance(group_count, int) or group_count <= 0:
        raise ValueError("roster must contain a positive parent clip count")
    if roster.get("verified_shard_count") != 184:
        raise ValueError("roster must cover the verified 184-shard metadata screen")

    sources = ledger.get("sources")
    if not isinstance(sources, list):
        raise ValueError("ledger sources must be a list")
    proposed: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in roster.get("sources", []):
        source_id = item.get("source_id")
        revision = item.get("revision")
        hashes = item.get("group_identity_sha256")
        if not isinstance(source_id, str) or not source_id.strip():
            raise ValueError("source ID is required")
        if not isinstance(revision, str) or not revision.strip():
            raise ValueError("source revision is required")
        key = (source_id, revision)
        if key in seen or any(
            source.get("source_id") == source_id and source.get("revision") == revision
            for source in sources
        ):
            raise ValueError(f"source revision already exists in ledger: {source_id}@{revision}")
        seen.add(key)
        if item.get("group_identity_kind") != "parent_clip_id":
            raise ValueError("unsupported source group identity kind")
        if item.get("group_count") != group_count:
            raise ValueError("source group count differs from the roster total")
        if (
            not isinstance(hashes, list)
            or len(hashes) != group_count
            or len(set(hashes)) != group_count
            or any(
                not isinstance(value, str) or not HEX_SHA256.fullmatch(value)
                for value in hashes
            )
        ):
            raise ValueError("group identity hashes must be unique lowercase SHA-256 values")

        proposed.append(
            {
                "coverage_note": (
                    "Known parent-clip IDs were exposed through third-party label metadata only. "
                    "No source media, transcript, or callsign text was read. "
                    "Broader cross-worktree "
                    "history and speaker/aircraft grouping remain unknown; all use stays blocked."
                ),
                "group_identity_kind": "parent_clip_id",
                "history_status": "INCOMPLETE",
                "revision": revision,
                "source_id": source_id,
                "usage": [
                    {
                        "cohort_role": "metadata_group_identity_exposure_no_model_fit",
                        "evidence": [{"path": roster_path, "sha256": roster_sha256.lower()}],
                        "group_count": group_count,
                        "group_identity_sha256": hashes,
                    }
                ],
            }
        )

    if len(proposed) != 3:
        raise ValueError("expected official, utterance-derivative, and label-derivative revisions")

    result = json.loads(json.dumps(ledger))
    result["generated_from"] = (
        "prior saved validation predictions plus a pinned metadata-only group exposure roster; "
        "no model or source media opened"
    )
    result["sources"].extend(proposed)
    result["sources"].sort(key=lambda source: (source["source_id"], source["revision"]))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--roster", required=True, type=Path)
    args = parser.parse_args()
    ledger = json.loads(args.ledger.read_text(encoding="utf-8"))
    roster_bytes = args.roster.read_bytes()
    roster = json.loads(roster_bytes)
    result = register_roster(
        ledger,
        roster,
        roster_path=str(args.roster),
        roster_sha256=hashlib.sha256(roster_bytes).hexdigest(),
    )
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    temporary = args.ledger.with_suffix(args.ledger.suffix + ".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(args.ledger)
    print(
        json.dumps(
            {
                "ledger": str(args.ledger),
                "source_count": len(result["sources"]),
                "registered_group_count_per_source": roster["unique_parent_clip_count"],
                "history_status": "INCOMPLETE",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
