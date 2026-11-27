"""Freeze a deterministic, near-duplicate-component-aware BANKING77 train split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from scripts.audit_banking77_train_quality import (
    EXPECTED_SHA256,
    audit_file,
    near_duplicate_candidates,
    normalize_text,
)


def _groups(rows: list[dict[str, str]]) -> list[list[int]]:
    parent: dict[int, int] = {}

    def find(row_id: int) -> int:
        parent.setdefault(row_id, row_id)
        if parent[row_id] != row_id:
            parent[row_id] = find(parent[row_id])
        return parent[row_id]

    pairs = near_duplicate_candidates(rows)
    for pair in pairs:
        left, right = int(pair["left_row_id"]), int(pair["right_row_id"])
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    exact_text_rows: dict[str, list[int]] = defaultdict(list)
    for row_id, row in enumerate(rows):
        exact_text_rows[normalize_text(row["text"])].append(row_id)
    for row_ids in exact_text_rows.values():
        if len(row_ids) > 1:
            root = find(row_ids[0])
            for row_id in row_ids[1:]:
                other = find(row_id)
                if root != other:
                    parent[other] = root

    by_root: dict[int, list[int]] = defaultdict(list)
    for row_id in range(len(rows)):
        by_root[find(row_id)].append(row_id)
    return list(by_root.values())


def _split_groups(
    rows: list[dict[str, str]], groups: list[list[int]], *, seed: int, validation_fraction: float
) -> set[int]:
    labels = sorted({row["category"] for row in rows})
    total_by_label = Counter(row["category"] for row in rows)
    target_by_label = {
        label: validation_fraction * total_by_label[label] for label in labels
    }
    target_total = validation_fraction * len(rows)

    rng = random.Random(seed)
    shuffled = list(groups)
    rng.shuffle(shuffled)
    shuffled.sort(key=len, reverse=True)
    selected: set[int] = set()
    validation_counts: Counter[str] = Counter()

    def objective(counts: Counter[str], total: int) -> float:
        class_error = sum(
            ((counts[label] - target_by_label[label]) ** 2)
            / max(target_by_label[label], 1.0)
            for label in labels
        )
        total_error = ((total - target_total) ** 2) / max(target_total, 1.0)
        return class_error + total_error

    validation_count = 0
    for group in shuffled:
        group_counts = Counter(rows[row_id]["category"] for row_id in group)
        proposed = validation_counts + group_counts
        if objective(proposed, validation_count + len(group)) < objective(
            validation_counts, validation_count
        ):
            selected.update(group)
            validation_counts = proposed
            validation_count += len(group)

    # Ensure every class remains represented on both sides, moving only whole groups.
    row_to_group = {row_id: group for group in groups for row_id in group}
    for label in labels:
        if not any(rows[row_id]["category"] == label for row_id in selected):
            candidates = [
                group
                for group in groups
                if group[0] not in selected
                and any(rows[row_id]["category"] == label for row_id in group)
            ]
            if not candidates:
                raise ValueError(f"cannot put intent {label!r} in validation")
            group = min(candidates, key=lambda item: (len(item), min(item)))
            selected.update(group)
        if not any(
            rows[row_id]["category"] == label
            for row_id in range(len(rows))
            if row_id not in selected
        ):
            candidates = [
                group
                for group in groups
                if group[0] in selected
                and any(rows[row_id]["category"] == label for row_id in group)
                and any(rows[row_id]["category"] != label for row_id in group)
            ]
            if not candidates:
                raise ValueError(f"cannot keep intent {label!r} in training")
            group = min(candidates, key=lambda item: (len(item), min(item)))
            selected.difference_update(group)
    del row_to_group
    return selected


def freeze_split(
    train_csv: Path,
    *,
    expected_sha256: str = EXPECTED_SHA256,
    seed: int = 17,
    validation_fraction: float = 0.20,
) -> dict[str, Any]:
    audit = audit_file(train_csv, expected_sha256)
    rows = list(csv.DictReader(io.StringIO(train_csv.read_text(encoding="utf-8-sig"), newline="")))
    groups = _groups(rows)
    validation_ids = _split_groups(
        rows, groups, seed=seed, validation_fraction=validation_fraction
    )
    train_ids = set(range(len(rows))) - validation_ids
    id_to_group = {row_id: group_id for group_id, group in enumerate(groups) for row_id in group}
    train_groups = {id_to_group[row_id] for row_id in train_ids}
    validation_groups = {id_to_group[row_id] for row_id in validation_ids}
    if train_groups & validation_groups:
        raise ValueError("candidate component crosses train/validation")
    train_counts = Counter(rows[row_id]["category"] for row_id in train_ids)
    validation_counts = Counter(rows[row_id]["category"] for row_id in validation_ids)
    if set(train_counts) != set(validation_counts):
        raise ValueError("every intent must occur in both train and validation")

    pair_summary = audit["audit"]["near_duplicate_screen"]["connected_component_summary"]
    ordered_train = sorted(train_ids)
    ordered_validation = sorted(validation_ids)
    validation_id_hash = hashlib.sha256(
        ("\n".join(map(str, ordered_validation)) + "\n").encode("ascii")
    ).hexdigest()
    train_id_hash = hashlib.sha256(
        ("\n".join(map(str, ordered_train)) + "\n").encode("ascii")
    ).hexdigest()
    return {
        "schema_version": 1,
        "source": audit["source"],
        "source_revision": audit["source_revision"],
        "source_sha256": audit["input_sha256"],
        "split_seed": seed,
        "validation_fraction_target": validation_fraction,
        "split_method": (
            "greedy class-balanced whole-group allocation over connected components "
            "of the pinned train-only lexical candidate graph; exact normalized "
            "duplicates are joined"
        ),
        "candidate_graph": pair_summary,
        "train_row_count": len(train_ids),
        "validation_row_count": len(validation_ids),
        "train_row_id_sha256": train_id_hash,
        "validation_row_id_sha256": validation_id_hash,
        "train_row_ids": ordered_train,
        "validation_row_ids": ordered_validation,
        "train_intent_counts": dict(sorted(train_counts.items())),
        "validation_intent_counts": dict(sorted(validation_counts.items())),
        "validation_order": "ascending source train row index",
        "all_intents_in_both_splits": True,
        "candidate_component_overlap": 0,
        "publisher_test_opened": False,
        "sealed_audit_opened": False,
        "limitations": [
            "heuristic lexical components are leakage proxies, not user/session identity groups",
            "the source has one known exact normalized conflicting-label pair; "
            "labels are unchanged",
            "the validation split is development data derived only from publisher train",
            "allocation is deterministic but greedy, not an optimal stratified partition",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train_csv", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--expected-sha256", default=EXPECTED_SHA256)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    args = parser.parse_args()
    result = freeze_split(
        args.train_csv,
        expected_sha256=args.expected_sha256,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
