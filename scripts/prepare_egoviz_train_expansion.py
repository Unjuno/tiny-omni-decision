"""Select unused publisher sessions for a training-only EgoViz expansion."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

INDEX_SHA256 = "c83126637cc24a7ac2b3600724b50cf9c53966b9b098acc57d5534e14ade1074"
PREVIOUS_PLAN_SHA256 = "8f19a087ab0d63031ce28045def2767acfe03290951da0f4b2abfc327428672c"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_expansion(records, excluded_groups, *, count=24, seed=20261030):
    eligible = [
        dict(row)
        for row in records
        if str(row["group_id"]) not in excluded_groups
        and float(row["duration_s"]) >= 15
        and 0 < int(row["mcap_bytes"]) <= 1_000_000_000
    ]
    if len({str(row["group_id"]) for row in eligible}) < count:
        raise ValueError("not enough distinct eligible sessions")
    if len({str(row["chunk_id"]) for row in eligible}) != len(eligible):
        raise ValueError("duplicate eligible chunk identity")
    categories = sorted(
        {str(row["category"]) for row in eligible},
        key=lambda category: (
            len({str(r["group_id"]) for r in eligible if r["category"] == category}),
            category,
        ),
    )
    if count < 2 * len(categories):
        raise ValueError("capacity is too small for two sessions per category")
    selected, used = [], set()

    def pick(category):
        options = [
            row
            for row in eligible
            if row["category"] == category and str(row["group_id"]) not in used
        ]
        options.sort(
            key=lambda row: (
                int(row["mcap_bytes"]),
                hashlib.sha256(f"{seed}:{row['group_id']}:{row['chunk_id']}".encode()).hexdigest(),
            )
        )
        if not options:
            return False
        chosen = options[0]
        selected.append(chosen)
        used.add(str(chosen["group_id"]))
        return True

    for _ in range(2):
        for category in categories:
            if not pick(category):
                raise ValueError(f"category {category} lacks distinct sessions")
    while len(selected) < count:
        choices = sorted(
            categories,
            key=lambda category: (
                sum(row["category"] == category for row in selected),
                categories.index(category),
            ),
        )
        if not any(pick(category) for category in choices):
            raise ValueError("not enough distinct eligible sessions")
    return selected


def build(index: Path, previous_plan: Path):
    import pyarrow.parquet as parquet

    if sha(index) != INDEX_SHA256 or sha(previous_plan) != PREVIOUS_PLAN_SHA256:
        raise ValueError("index or prior frozen plan hash mismatch")
    prior = json.loads(previous_plan.read_text(encoding="utf-8"))
    excluded = set(prior["source"]["previous_groups_excluded"])
    excluded.update(str(row["group_id"]) for rows in prior["splits"].values() for row in rows)
    selected = select_expansion(parquet.read_table(index).to_pylist(), excluded)
    train = [
        {
            "chunk_id": str(row["chunk_id"]),
            "group_id": str(row["group_id"]),
            "category": str(row["category"]),
            "duration_s": float(row["duration_s"]),
            "expected_bytes": int(row["mcap_bytes"]),
            "mcap_path": str(row["mcap_path"]),
        }
        for row in selected
    ]
    return {
        "format": "egoviz-unseen-session-temporal-correspondence-plan-v1",
        "status": "METADATA_ONLY_TRAIN_EXPANSION",
        "source": {
            "dataset": "humaidtech/EgoViz-120",
            "license": "CC-BY-4.0",
            "index_sha256": INDEX_SHA256,
            "previous_plan_sha256": PREVIOUS_PLAN_SHA256,
            "excluded_group_count": len(excluded),
            "excluded_groups_sha256": hashlib.sha256(
                "\n".join(sorted(excluded)).encode()
            ).hexdigest(),
            "seed": 20261030,
            "selection": (
                "two per available category, rare categories first, then balanced; "
                "smallest files with seeded tie break"
            ),
        },
        "splits": {"train": train, "development": [], "sealed_audit": []},
        "counts": {
            "train_groups": len(train),
            "expected_bytes": sum(r["expected_bytes"] for r in train),
            "category_counts": {
                c: sum(r["category"] == c for r in train)
                for c in sorted({r["category"] for r in train})
            },
        },
        "previous_reserved_groups_overlap": 0,
        "sealed_media_accessed": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--previous-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = build(args.index, args.previous_plan)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(plan, stream, indent=2)
        stream.write("\n")
    print(json.dumps(plan["counts"], indent=2))


if __name__ == "__main__":
    main()
