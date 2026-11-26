"""Hash and compare only the ATOMIC 2020 train/dev TSV files.

This utility deliberately has no test-split input. It reports counts and hashes
without emitting source text, targets, or model predictions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path


def _normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_rows(
    path: Path,
) -> tuple[list[tuple[str, str, str]], dict[str, object], set[str]]:
    rows: list[tuple[str, str, str]] = []
    all_width_three_heads: set[str] = set()
    input_rows = 0
    malformed_width_rows = 0
    empty_field_rows = 0
    empty_field_patterns: Counter[str] = Counter()
    with path.open(encoding="utf-8-sig", newline="") as source:
        for row_number, row in enumerate(csv.reader(source, delimiter="\t"), 1):
            input_rows = row_number
            if len(row) != 3:
                malformed_width_rows += 1
                continue
            all_width_three_heads.add(row[0])
            empty_positions = tuple(
                str(index) for index, value in enumerate(row) if not value.strip()
            )
            if empty_positions:
                empty_field_rows += 1
                empty_field_patterns[",".join(empty_positions)] += 1
                continue
            rows.append((row[0], row[1], row[2]))
    if not rows:
        raise ValueError(f"{path}: no valid three-field rows")
    return rows, {
        "input_rows": input_rows,
        "valid_rows": len(rows),
        "malformed_width_rows": malformed_width_rows,
        "empty_field_rows": empty_field_rows,
        "empty_field_position_counts_zero_based": dict(sorted(empty_field_patterns.items())),
    }, all_width_three_heads


def _triple_sets(
    rows: Iterable[tuple[str, str, str]],
) -> tuple[set[tuple[str, str, str]], set[tuple[str, str, str]]]:
    exact = set(rows)
    normalized = {
        (_normalize(head), _normalize(relation), _normalize(tail))
        for head, relation, tail in rows
    }
    return exact, normalized


def _split_summary(
    path: Path,
    rows: list[tuple[str, str, str]],
    row_quality: dict[str, object],
    all_width_three_heads: set[str],
) -> dict[str, object]:
    normalized_heads = {_normalize(head) for head in all_width_three_heads}
    normalized_tails_by_pair: dict[tuple[str, str], set[str]] = defaultdict(set)
    relation_counts = Counter(relation for _, relation, _ in rows)
    for head, relation, tail in rows:
        normalized_tails_by_pair[(_normalize(head), _normalize(relation))].add(
            _normalize(tail)
        )
    tail_cardinality = Counter(len(tails) for tails in normalized_tails_by_pair.values())
    exact_triples, normalized_triples = _triple_sets(rows)

    return {
        "path_name": path.name,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "rows": row_quality["input_rows"],
        "valid_rows": len(rows),
        "malformed_width_rows": row_quality["malformed_width_rows"],
        "empty_field_rows": row_quality["empty_field_rows"],
        "empty_field_position_counts_zero_based": row_quality[
            "empty_field_position_counts_zero_based"
        ],
        "unique_exact_triples": len(exact_triples),
        "exact_duplicate_rows": len(rows) - len(exact_triples),
        "unique_normalized_triples": len(normalized_triples),
        "normalized_duplicate_rows": len(rows) - len(normalized_triples),
        "normalized_event_heads": len(normalized_heads),
        "normalized_head_relation_pairs": len(normalized_tails_by_pair),
        "head_relation_pairs_with_one_tail": tail_cardinality[1],
        "head_relation_pairs_with_multiple_tails": sum(
            count for cardinality, count in tail_cardinality.items() if cardinality > 1
        ),
        "relation_counts": dict(sorted(relation_counts.items())),
        "none_tail_rows": sum(_normalize(tail) == "none" for _, _, tail in rows),
    }


def audit_train_dev(train_path: Path, dev_path: Path) -> dict[str, object]:
    train, train_quality, train_raw_heads = _read_rows(train_path)
    dev, dev_quality, dev_raw_heads = _read_rows(dev_path)
    train_exact, train_normalized = _triple_sets(train)
    dev_exact, dev_normalized = _triple_sets(dev)
    train_normalized_heads = {_normalize(head) for head in train_raw_heads}
    dev_normalized_heads = {_normalize(head) for head in dev_raw_heads}

    return {
        "scope": "train_and_dev_only; no test-split argument or read",
        "train": _split_summary(
            train_path, train, train_quality, train_raw_heads
        ),
        "dev": _split_summary(dev_path, dev, dev_quality, dev_raw_heads),
        "cross_split": {
            "raw_event_head_overlap": len(train_raw_heads & dev_raw_heads),
            "normalized_event_head_overlap": len(
                train_normalized_heads & dev_normalized_heads
            ),
            "normalized_dev_heads": len(dev_normalized_heads),
            "normalized_dev_head_overlap_fraction": (
                len(train_normalized_heads & dev_normalized_heads)
                / len(dev_normalized_heads)
            ),
            "exact_triple_overlap": len(train_exact & dev_exact),
            "normalized_triple_overlap": len(train_normalized & dev_normalized),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit_train_dev(args.train, args.dev), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
