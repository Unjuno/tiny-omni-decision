"""Create a deterministic, session-disjoint metadata plan for a fresh EgoViz probe.

The planner reads only the public clip index and the previous acquisition manifest.
It does not retrieve media, read action labels, or load sealed-audit media.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

DEFAULT_CAPACITIES = {"train": 24, "development": 20, "sealed_audit": 20}
DEFAULT_MAX_MCAP_BYTES = 1_000_000_000
MIN_CLIP_DURATION_SECONDS = 15.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tie_key(seed: int, split: str, category: str, group_id: str) -> str:
    value = f"{seed}:{split}:{category}:{group_id}".encode()
    return hashlib.sha256(value).hexdigest()


def select_grouped_records(
    records: list[dict[str, Any]],
    excluded_groups: set[str],
    capacities: dict[str, int] | None = None,
    *,
    seed: int = 20261029,
    max_mcap_bytes: int = DEFAULT_MAX_MCAP_BYTES,
) -> dict[str, list[dict[str, Any]]]:
    """Select one clip per unseen session, balancing activity categories by split."""
    split_sizes = capacities or DEFAULT_CAPACITIES
    split_names = tuple(split_sizes)
    if split_names != ("train", "development", "sealed_audit"):
        raise ValueError("splits must be train, development, sealed_audit in that order")
    if any(size <= 0 for size in split_sizes.values()):
        raise ValueError("split capacities must be positive")

    eligible = [
        dict(row)
        for row in records
        if str(row["group_id"]) not in excluded_groups
        and float(row["duration_s"]) >= MIN_CLIP_DURATION_SECONDS
        and 0 < int(row["mcap_bytes"]) <= max_mcap_bytes
    ]
    if not eligible:
        raise ValueError("no eligible unseen records")
    chunk_ids = [str(row["chunk_id"]) for row in eligible]
    if len(set(chunk_ids)) != len(chunk_ids):
        raise ValueError("eligible index contains duplicate chunk IDs")

    by_category: dict[str, list[dict[str, Any]]] = {}
    for row in eligible:
        by_category.setdefault(str(row["category"]), []).append(row)
    category_groups = {
        category: {str(row["group_id"]) for row in rows}
        for category, rows in by_category.items()
    }
    category_order = sorted(
        by_category,
        key=lambda category: (len(category_groups[category]), category),
    )
    for category in category_order:
        if len(category_groups[category]) < len(split_names):
            raise ValueError(
                f"category {category!r} needs distinct eligible groups in each split"
            )
        if any(split_sizes[split] < 1 for split in split_names):
            raise ValueError(f"split capacity cannot cover category {category!r}")

    selected: dict[str, list[dict[str, Any]]] = {split: [] for split in split_names}
    group_assignment: dict[str, str] = {}

    def candidates(category: str, split: str) -> list[dict[str, Any]]:
        rows = [
            row
            for row in by_category[category]
            if str(row["group_id"]) not in group_assignment
        ]
        return sorted(
            rows,
            key=lambda row: (
                int(row["mcap_bytes"]),
                _tie_key(seed, split, category, str(row["group_id"])),
                str(row["chunk_id"]),
            ),
        )

    # Reserve a distinct group for every category in every split before filling
    # remaining capacity. The sealed membership is metadata-only at this stage.
    for category in category_order:
        for split in split_names:
            if len(selected[split]) >= split_sizes[split]:
                raise ValueError(f"capacity too small for category coverage in {split}")
            options = candidates(category, split)
            if not options:
                raise ValueError(
                    f"category {category!r} cannot be represented by distinct eligible groups"
                )
            chosen = options[0]
            group_id = str(chosen["group_id"])
            group_assignment[group_id] = split
            selected[split].append(chosen)

    for split in split_names:
        while len(selected[split]) < split_sizes[split]:
            category_counts = {
                category: sum(str(row["category"]) == category for row in selected[split])
                for category in by_category
            }
            categories = sorted(
                by_category,
                key=lambda category: (
                    category_counts[category],
                    len(category_groups[category]),
                    category,
                ),
            )
            chosen = None
            for category in categories:
                options = candidates(category, split)
                if options:
                    chosen = options[0]
                    break
            if chosen is None:
                raise ValueError("not enough distinct eligible groups to fill split plan")
            group_id = str(chosen["group_id"])
            group_assignment[group_id] = split
            selected[split].append(chosen)

    for split in split_names:
        selected[split].sort(key=lambda row: (str(row["category"]), str(row["chunk_id"])))
    return selected


def build_plan(
    index_path: Path,
    previous_acquisition_path: Path,
    *,
    seed: int = 20261029,
    capacities: dict[str, int] | None = None,
    max_mcap_bytes: int = DEFAULT_MAX_MCAP_BYTES,
) -> dict[str, Any]:
    """Build a hash-pinned metadata-only acquisition plan from EgoViz index parquet."""
    import pyarrow.parquet as parquet

    table = parquet.read_table(index_path)
    records = table.to_pylist()
    previous = json.loads(previous_acquisition_path.read_text(encoding="utf-8"))
    previous_files = previous.get("files", [])
    excluded_groups = {str(row["group_id"]) for row in previous_files}
    selected = select_grouped_records(
        records,
        excluded_groups,
        capacities,
        seed=seed,
        max_mcap_bytes=max_mcap_bytes,
    )
    flattened = [row for split_rows in selected.values() for row in split_rows]
    group_sets = {
        split: {str(row["group_id"]) for row in rows}
        for split, rows in selected.items()
    }
    for left_index, left in enumerate(group_sets):
        for right in list(group_sets)[left_index + 1 :]:
            if group_sets[left] & group_sets[right]:
                raise ValueError(f"group overlap between {left} and {right}")

    splits: dict[str, list[dict[str, Any]]] = {}
    for split, rows in selected.items():
        splits[split] = [
            {
                "chunk_id": str(row["chunk_id"]),
                "group_id": str(row["group_id"]),
                "category": str(row["category"]),
                "duration_s": float(row["duration_s"]),
                "expected_bytes": int(row["mcap_bytes"]),
                "mcap_path": str(row["mcap_path"]),
            }
            for row in rows
        ]

    return {
        "format": "egoviz-unseen-session-temporal-correspondence-plan-v1",
        "status": "METADATA_ONLY_PLAN_MEDIA_NOT_DOWNLOADED",
        "source": {
            "dataset": "humaidtech/EgoViz-120",
            "bucket": "hf://buckets/humaidtech/EgoViz-120",
            "license": "CC-BY-4.0, as stated by the official dataset card",
            "index_sha256": sha256_file(index_path),
            "previous_acquisition_sha256": sha256_file(previous_acquisition_path),
            "previous_groups_excluded": sorted(excluded_groups),
            "split_unit": (
                "publisher group_id session; participant independence across "
                "sessions is unknown"
            ),
            "seed": seed,
            "selection_rule": (
                "exclude every previously used group; require duration >= 15s and "
                f"MCAP size <= {max_mcap_bytes} bytes; reserve one smallest eligible "
                "file per category and split, then fill to fixed group capacities by "
                "least represented category with smallest eligible file and SHA tie-break"
            ),
        },
        "task": {
            "name": "four-choice video-to-IMU timestamp correspondence",
            "inputs": ["8 wrist-left video frames", "6-channel wrist-left IMU window"],
            "duration_seconds": 2,
            "stride_seconds": 1,
            "candidates": 4,
            "negative_windows": "same clip, at least 3 seconds from the matching timestamp",
            "label_source": (
                "exact shared-clock interval identity; publisher /action labels "
                "are not used"
            ),
            "question": (
                "Which IMU option comes from the same two-second interval as "
                "this video window?"
            ),
        },
        "split_counts": {
            split: {
                "unique_groups": len(rows),
                "unique_clips": len({str(row["chunk_id"]) for row in rows}),
                "estimated_bytes": sum(int(row["mcap_bytes"]) for row in rows),
                "category_counts": {
                    category: sum(str(row["category"]) == category for row in rows)
                    for category in sorted({str(row["category"]) for row in rows})
                },
            }
            for split, rows in selected.items()
        },
        "totals": {
            "selected_groups": len(flattened),
            "selected_bytes": sum(int(row["mcap_bytes"]) for row in flattened),
            "train_development_bytes": sum(
                int(row["mcap_bytes"])
                for split in ("train", "development")
                for row in selected[split]
            ),
            "largest_file_bytes": max(int(row["mcap_bytes"]) for row in flattened),
            "previous_split_group_overlap": 0,
            "new_split_group_overlap": 0,
            "sealed_audit_media_downloaded": False,
            "sealed_audit_media_loaded": False,
            "sealed_audit_evaluated": False,
            "publisher_action_labels_read": False,
        },
        "splits": splits,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--previous-acquisition", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261029)
    parser.add_argument("--max-mcap-bytes", type=int, default=DEFAULT_MAX_MCAP_BYTES)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    plan = build_plan(
        args.index,
        args.previous_acquisition,
        seed=args.seed,
        max_mcap_bytes=args.max_mcap_bytes,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: plan[key] for key in ("status", "split_counts", "totals")}, indent=2))


if __name__ == "__main__":
    main()
