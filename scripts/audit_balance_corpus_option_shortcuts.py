"""Audit Balance Corpus metadata for pair-group and option-set shortcuts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

REQUIRED_COLUMNS = {
    "pair_id",
    "clue_giver_condition",
    "target_word",
    "taboo_1",
    "taboo_2",
    "taboo_3",
    "taboo_4",
    "taboo_5",
}


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().casefold())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_metadata(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = set(reader.fieldnames or ())
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(f"metadata is missing required columns: {sorted(missing)}")
        rows = list(reader)

    if not rows:
        raise ValueError("metadata contains no rows")

    pair_counts: Counter[str] = Counter()
    condition_counts: Counter[str] = Counter()
    option_set_pairs: dict[tuple[str, ...], set[str]] = defaultdict(set)
    option_set_targets: dict[tuple[str, ...], set[str]] = defaultdict(set)
    target_pairs: dict[str, set[str]] = defaultdict(set)
    duplicate_option_rows = 0
    empty_option_rows = 0

    for row in rows:
        pair = row["pair_id"].strip()
        condition = _normalize(row["clue_giver_condition"])
        options = [
            _normalize(row[name])
            for name in (
                "target_word",
                "taboo_1",
                "taboo_2",
                "taboo_3",
                "taboo_4",
                "taboo_5",
            )
        ]
        if not pair or not all(options):
            empty_option_rows += 1
            continue
        if len(set(options)) != len(options):
            duplicate_option_rows += 1
        option_key = tuple(sorted(options))
        target = options[0]
        pair_counts[pair] += 1
        condition_counts[condition] += 1
        option_set_pairs[option_key].add(pair)
        option_set_targets[option_key].add(target)
        target_pairs[target].add(pair)

    if empty_option_rows:
        raise ValueError(f"metadata has {empty_option_rows} rows with empty pair/options")

    deterministic_option_sets = {
        option_key for option_key, targets in option_set_targets.items() if len(targets) == 1
    }
    repeated_deterministic_sets = {
        option_key
        for option_key in deterministic_option_sets
        if len(option_set_pairs[option_key]) > 1
    }

    return {
        "source_metadata_sha256": _sha256(path),
        "rows": len(rows),
        "pair_groups": len(pair_counts),
        "trials_per_pair_min": min(pair_counts.values()),
        "trials_per_pair_max": max(pair_counts.values()),
        "condition_counts": dict(sorted(condition_counts.items())),
        "unique_unordered_option_sets": len(option_set_pairs),
        "option_sets_shared_across_pair_groups": sum(
            len(pairs) > 1 for pairs in option_set_pairs.values()
        ),
        "deterministic_option_sets": len(deterministic_option_sets),
        "deterministic_option_sets_shared_across_pair_groups": len(repeated_deterministic_sets),
        "unique_target_words": len(target_pairs),
        "targets_reused_across_pair_groups": sum(len(pairs) > 1 for pairs in target_pairs.values()),
        "rows_with_duplicate_options": duplicate_option_rows,
        "pair_group_split_has_exact_option_set_shortcut": bool(repeated_deterministic_sets),
        "identifiers_or_raw_text_included": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = audit_metadata(args.metadata_csv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
