"""Build a hash-only roster of TartanAviation groups exposed in label metadata.

The reader accesses only the existing Parquet ``utterance_id`` column. It does
not read transcripts, callsign text, labels, or media. The resulting source-
scoped group digests can be used by the fail-closed cohort history gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

TOKEN_SPLIT = re.compile(r"[_:/-]+")
OFFICIAL_SOURCE = "castacks/TartanAviation"
OFFICIAL_REVISION = "077a04abac1e50953c45b72524edd79e3a21d9bd"
DERIVATIVE_SOURCE = "twangodev/tartanaviation-atc-labels"
DERIVATIVE_REVISION = "0775355c388aae64335522f40bff9fc5129b2922"
UTTERANCE_SOURCE = "twangodev/tartanaviation-atc-adsb-utterances"
UTTERANCE_REVISION = "bc588362d2f877e6ba0910fe7b6e035649d43918"


def canonical_clip_group(utterance_id: str) -> str:
    """Normalize {airport}/{MM-DD-YY}/{clip}/{utterance} to the clip group."""
    parts = TOKEN_SPLIT.split(utterance_id.strip())
    if len(parts) != 6 or any(not part for part in parts):
        raise ValueError("utterance_id must have six nonempty identity tokens")
    airport, month, day, year, clip_index, _utterance_index = parts
    return f"{airport.lower()}/{month}-{day}-{year}/{clip_index}"


def source_group_digest(source_id: str, revision: str, group_id: str) -> str:
    canonical = json.dumps(
        [source_id.strip(), revision.strip(), group_id.strip()],
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_exposure_roster(
    utterance_ids: list[str], *, summary_sha256: str, shard_sha256s: dict[str, str]
) -> dict[str, Any]:
    if not utterance_ids or len(utterance_ids) != len(set(utterance_ids)):
        raise ValueError("utterance IDs must be nonempty and unique")
    groups = sorted({canonical_clip_group(value) for value in utterance_ids})
    by_source = []
    for source_id, revision in (
        (OFFICIAL_SOURCE, OFFICIAL_REVISION),
        (DERIVATIVE_SOURCE, DERIVATIVE_REVISION),
        (UTTERANCE_SOURCE, UTTERANCE_REVISION),
    ):
        by_source.append(
            {
                "source_id": source_id,
                "revision": revision,
                "group_identity_kind": "parent_clip_id",
                "group_count": len(groups),
                "group_identity_sha256": [
                    source_group_digest(source_id, revision, group) for group in groups
                ],
            }
        )
    return {
        "schema_version": 1,
        "exposure": "metadata_only_utterance_ids_read_no_media_or_content",
        "metadata_summary_sha256": summary_sha256,
        "verified_shard_count": len(shard_sha256s),
        "verified_shard_hashes_sha256": hashlib.sha256(
            json.dumps(shard_sha256s, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "utterance_id_count": len(utterance_ids),
        "unique_parent_clip_count": len(groups),
        "group_identity_hash_encoding": (
            "SHA-256(canonical JSON [source_id, revision, normalized parent_clip_id])"
        ),
        "sources": by_source,
    }


def read_existing_metadata(shard_dir: Path, summary_path: Path) -> dict[str, Any]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("PyArrow is required to read the existing metadata shards") from exc

    summary_bytes = summary_path.read_bytes()
    summary_sha256 = hashlib.sha256(summary_bytes).hexdigest()
    summary = json.loads(summary_bytes)
    if summary.get("expected_shards") != 184 or summary.get("verified_shards") != 184:
        raise ValueError("metadata receipt does not verify the expected 184 shards")
    shard_receipts = {item["path"]: item["sha256"] for item in summary["shards"]}
    paths = sorted(shard_dir.glob("shard-*.parquet"))
    if len(paths) != 184 or set(path.name for path in paths) != set(shard_receipts):
        raise ValueError("local shard file set does not match the verified receipt")

    utterance_ids: list[str] = []
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != shard_receipts[path.name]:
            raise ValueError(f"shard hash mismatch: {path.name}")
        table = pq.read_table(path, columns=["utterance_id"])
        utterance_ids.extend(str(value) for value in table.column("utterance_id").to_pylist())
    if len(utterance_ids) != summary.get("rows"):
        raise ValueError("utterance ID count does not match the verified metadata receipt")

    return build_exposure_roster(
        utterance_ids,
        summary_sha256=summary_sha256,
        shard_sha256s=shard_receipts,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = read_existing_metadata(args.shard_dir, args.summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "verified_shard_count": result["verified_shard_count"],
                "utterance_id_count": result["utterance_id_count"],
                "unique_parent_clip_count": result["unique_parent_clip_count"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
