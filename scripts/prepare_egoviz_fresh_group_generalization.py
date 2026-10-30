"""Freeze a new group-disjoint EgoViz development set without reading labels/media."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as parquet

MIN_DURATION_SECONDS = 15.0
MAX_MCAP_BYTES = 1_000_000_000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or not isinstance(row.get("group_id"), str):
            raise ValueError(f"invalid sample identity in {path}")
        rows.append(row)
    return rows


def select_unseen_group_rows(
    records: list[dict[str, Any]], excluded_groups: set[str], *, seed: int
) -> list[dict[str, Any]]:
    """Retain every eligible unused group and choose one clip by a fixed hash key."""
    chunk_ids = [str(row["chunk_id"]) for row in records]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("duplicate chunk IDs in source index")

    by_group: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        group_id = str(row["group_id"])
        if group_id in excluded_groups:
            continue
        if (
            float(row["duration_s"]) < MIN_DURATION_SECONDS
            or int(row["mcap_bytes"]) <= 0
            or int(row["mcap_bytes"]) > MAX_MCAP_BYTES
        ):
            continue
        if not str(row["mcap_path"]).startswith("data/"):
            raise ValueError(f"invalid bucket-relative path for {row['chunk_id']}")
        by_group.setdefault(group_id, []).append(row)

    if not by_group:
        raise ValueError("no eligible unseen groups")

    selected = []
    for rows in by_group.values():
        selected.append(
            min(
                rows,
                key=lambda row: hashlib.sha256(
                    f"{seed}:egoviz-fresh-generalization:{row['chunk_id']}".encode()
                ).hexdigest(),
            )
        )
    return sorted(selected, key=lambda row: (str(row["category"]), str(row["group_id"])))


def build_plan(
    index_path: Path,
    previous_plan_path: Path,
    existing_corpus: Path,
    *,
    seed: int = 20261030,
) -> dict[str, Any]:
    previous_plan = json.loads(previous_plan_path.read_text(encoding="utf-8"))
    if previous_plan.get("format") != "egoviz-unseen-session-temporal-correspondence-plan-v1":
        raise ValueError("unsupported previous EgoViz plan")

    observed_groups = {
        row["group_id"]
        for split in ("train", "development")
        for row in _read_rows(existing_corpus / f"{split}-samples.jsonl")
    }
    prior_source_groups = set(previous_plan["source"].get("previous_groups_excluded", []))
    prior_train_dev_groups = {
        str(row["group_id"])
        for split in ("train", "development")
        for row in previous_plan["splits"][split]
    }
    reserved_sealed_groups = {
        str(row["group_id"]) for row in previous_plan["splits"]["sealed_audit"]
    }
    excluded = (
        observed_groups
        | prior_source_groups
        | prior_train_dev_groups
        | reserved_sealed_groups
    )

    table = parquet.read_table(
        index_path,
        columns=["chunk_id", "group_id", "category", "duration_s", "mcap_bytes", "mcap_path"],
    )
    records = table.to_pylist()
    selected = select_unseen_group_rows(records, excluded, seed=seed)
    selected_groups = {str(row["group_id"]) for row in selected}
    if selected_groups & excluded:
        raise ValueError("fresh development groups overlap prior/reserved groups")

    rows = [
        {
            "chunk_id": str(row["chunk_id"]),
            "group_id": str(row["group_id"]),
            "category": str(row["category"]),
            "duration_s": float(row["duration_s"]),
            "expected_bytes": int(row["mcap_bytes"]),
            "mcap_path": str(row["mcap_path"]),
            "split": "development",
        }
        for row in selected
    ]
    category_counts: dict[str, int] = {}
    for row in rows:
        category_counts[row["category"]] = category_counts.get(row["category"], 0) + 1
    selected_bytes = sum(row["expected_bytes"] for row in rows)
    return {
        "format": "egoviz-unseen-session-temporal-correspondence-plan-v1",
        "status": "METADATA_ONLY_FRESH_DEVELOPMENT_PLAN_MEDIA_NOT_DOWNLOADED",
        "source": {
            "dataset": "humaidtech/EgoViz-120",
            "license": "CC-BY-4.0, as stated by the official dataset card",
            "index_sha256": sha256_file(index_path),
            "previous_plan_sha256": sha256_file(previous_plan_path),
            "existing_corpus_train_sha256": sha256_file(existing_corpus / "train-samples.jsonl"),
            "existing_corpus_development_sha256": sha256_file(
                existing_corpus / "development-samples.jsonl"
            ),
            "seed": seed,
            "split_unit": "publisher group_id session; participant independence remains unknown",
            "selection_rule": (
                "exclude every observed train/development group and every group reserved by the "
                "prior plan; include every remaining group with a >=15s clip <=1GB; select one "
                "eligible clip per group by SHA-256 ordering of seed and chunk ID"
            ),
            "excluded_groups": sorted(excluded),
            "reserved_prior_sealed_group_count": len(reserved_sealed_groups),
            "prior_sealed_media_accessed": False,
            "publisher_action_labels_read": False,
        },
        "task": previous_plan["task"],
        "split_counts": {
            "train": {"unique_groups": 0, "unique_clips": 0, "estimated_bytes": 0},
            "development": {
                "unique_groups": len(rows),
                "unique_clips": len(rows),
                "estimated_bytes": selected_bytes,
                "category_counts": category_counts,
            },
            "sealed_audit": {"unique_groups": 0, "unique_clips": 0, "estimated_bytes": 0},
        },
        "totals": {
            "observed_groups_excluded": len(observed_groups),
            "reserved_groups_excluded": len(excluded),
            "eligible_unseen_groups": len(rows),
            "selected_bytes": selected_bytes,
            "largest_selected_file_bytes": max(row["expected_bytes"] for row in rows),
            "train_development_group_overlap": 0,
            "prior_sealed_media_downloaded": False,
            "prior_sealed_media_loaded": False,
            "prior_sealed_media_evaluated": False,
            "publisher_action_labels_read": False,
        },
        "splits": {"train": [], "development": rows, "sealed_audit": []},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--previous-plan", type=Path, required=True)
    parser.add_argument("--existing-corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261030)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    plan = build_plan(args.index, args.previous_plan, args.existing_corpus, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": plan["status"],
                "groups": plan["totals"]["eligible_unseen_groups"],
                "bytes": plan["totals"]["selected_bytes"],
                "largest_file_bytes": plan["totals"]["largest_selected_file_bytes"],
                "category_counts": plan["split_counts"]["development"]["category_counts"],
                "plan": str(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
