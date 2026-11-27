"""Build a frozen CLINC150 dev split that groups lexical near-duplicates."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROTOCOL_ID = "clinc150-near-duplicate-grouped-development-v1"
TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _tokens(text: str) -> set[str]:
    return set(TOKEN_RE.findall(text.casefold()))


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while item != self.parent[item]:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        root_left, root_right = self.find(left), self.find(right)
        if root_left == root_right:
            return
        if self.rank[root_left] < self.rank[root_right]:
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        if self.rank[root_left] == self.rank[root_right]:
            self.rank[root_left] += 1


def build_near_duplicate_split(
    rows: list[dict[str, Any]],
    *,
    seed: int = 17,
    validation_fraction: float = 0.20,
    jaccard_threshold: float = 0.80,
    max_bigram_document_frequency: int = 80,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Group exact and lexical near duplicates before deterministic splitting.

    Candidate pairs share a sorted adjacent token bigram whose pooled document
    frequency is at most ``max_bigram_document_frequency``. Candidate pairs are
    joined when token-set Jaccard meets the fixed threshold. Transitive matches
    stay in one connected component, including cross-label conflicts.
    """
    if not rows:
        raise ValueError("rows must not be empty")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must lie strictly between zero and one")
    if not 0 < jaccard_threshold <= 1:
        raise ValueError("jaccard_threshold must lie in (0, 1]")
    if max_bigram_document_frequency < 1:
        raise ValueError("max_bigram_document_frequency must be positive")
    sample_ids = [str(row["sample_id"]) for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("sample_id values must be unique")

    normalized = [normalize_text(str(row["text"])) for row in rows]
    token_sets = [_tokens(text) for text in normalized]
    if any(not tokens for tokens in token_sets):
        raise ValueError("every row must contain at least one alphanumeric token")
    union_find = _UnionFind(len(rows))

    # Keep normalized exact-content duplicates together regardless of tokenization.
    exact_groups: dict[str, list[int]] = defaultdict(list)
    for index, text in enumerate(normalized):
        exact_groups[text].append(index)
    for members in exact_groups.values():
        for index in members[1:]:
            union_find.union(members[0], index)

    bigram_rows: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, tokens in enumerate(token_sets):
        ordered_tokens = sorted(tokens)
        for bigram in zip(ordered_tokens, ordered_tokens[1:], strict=False):
            bigram_rows[bigram].append(index)

    candidate_pairs: set[tuple[int, int]] = set()
    for members in bigram_rows.values():
        if len(members) > max_bigram_document_frequency:
            continue
        for position, left in enumerate(members):
            for right in members[position + 1 :]:
                candidate_pairs.add((left, right))

    near_duplicate_pairs = 0
    for left, right in candidate_pairs:
        left_tokens, right_tokens = token_sets[left], token_sets[right]
        similarity = len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
        if similarity >= jaccard_threshold:
            near_duplicate_pairs += 1
            union_find.union(left, right)

    components: dict[int, list[int]] = defaultdict(list)
    for index in range(len(rows)):
        components[union_find.find(index)].append(index)
    ordered_components = sorted(
        components.values(), key=lambda members: min(sample_ids[index] for index in members)
    )
    component_ids: list[str] = [""] * len(rows)
    conflicting_components = 0
    for members in ordered_components:
        member_ids = sorted(sample_ids[index] for index in members)
        component_id = sha256_bytes("\n".join(member_ids).encode("utf-8"))
        labels = {int(rows[index]["label"]) for index in members}
        conflicting_components += len(labels) > 1
        for index in members:
            component_ids[index] = component_id

    strata: dict[tuple[int, ...], list[str]] = defaultdict(list)
    component_members: dict[str, list[int]] = defaultdict(list)
    for index, component_id in enumerate(component_ids):
        component_members[component_id].append(index)
    for component_id, members in component_members.items():
        labels = tuple(sorted({int(rows[index]["label"]) for index in members}))
        strata[labels].append(component_id)

    validation_components: set[str] = set()
    stratum_receipts: dict[str, dict[str, int]] = {}
    for labels, component_ids_in_stratum in sorted(strata.items()):
        ordered = sorted(
            component_ids_in_stratum,
            key=lambda component_id: hashlib.sha256(
                f"{seed}:{PROTOCOL_ID}:{','.join(map(str, labels))}:{component_id}".encode()
            ).hexdigest(),
        )
        validation_count = int(round(len(ordered) * validation_fraction))
        if len(ordered) >= 2:
            validation_count = min(max(validation_count, 1), len(ordered) - 1)
        validation_components.update(ordered[:validation_count])
        stratum_receipts[",".join(map(str, labels))] = {
            "component_count": len(ordered),
            "validation_component_count": validation_count,
        }

    train_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    train_label_counts: Counter[int] = Counter()
    validation_label_counts: Counter[int] = Counter()
    for index, row in enumerate(rows):
        component_id = component_ids[index]
        output_row = {
            "sample_id": sample_ids[index],
            "near_duplicate_group_id": component_id,
            "text": row["text"],
            "label": int(row["label"]),
            "label_name": str(row["label_name"]),
        }
        if component_id in validation_components:
            validation_rows.append(output_row)
            validation_label_counts[int(row["label"])] += 1
        else:
            train_rows.append(output_row)
            train_label_counts[int(row["label"])] += 1
    train_rows.sort(key=lambda row: row["sample_id"])
    validation_rows.sort(key=lambda row: row["sample_id"])

    train_groups = {row["near_duplicate_group_id"] for row in train_rows}
    validation_groups = {row["near_duplicate_group_id"] for row in validation_rows}
    if train_groups & validation_groups:
        raise AssertionError("near-duplicate component leakage across splits")
    all_labels = set(train_label_counts) | set(validation_label_counts)
    if set(train_label_counts) != all_labels or set(validation_label_counts) != all_labels:
        raise ValueError("near-duplicate grouped split lost class coverage")

    receipt = {
        "protocol_id": PROTOCOL_ID,
        "seed": seed,
        "validation_fraction_target": validation_fraction,
        "group_rule": {
            "exact_normalization": "NFKC, casefold, collapsed whitespace",
            "candidate_generation": (
                "sorted adjacent unique-token bigram; pooled document frequency <= 80"
            ),
            "near_duplicate_rule": f"token-set Jaccard >= {jaccard_threshold:.2f}",
            "transitivity": "connected components; cross-label conflicts remain grouped",
        },
        "source_rows": len(rows),
        "near_duplicate_groups": len(components),
        "largest_group_rows": max(map(len, components.values())),
        "near_duplicate_pairs_examined": len(candidate_pairs),
        "near_duplicate_pairs_joined": near_duplicate_pairs,
        "cross_label_group_count": conflicting_components,
        "train_rows": len(train_rows),
        "train_groups": len(train_groups),
        "validation_rows": len(validation_rows),
        "validation_groups": len(validation_groups),
        "cross_split_group_overlap": len(train_groups & validation_groups),
        "train_class_counts": {str(key): train_label_counts[key] for key in sorted(all_labels)},
        "validation_class_counts": {
            str(key): validation_label_counts[key] for key in sorted(all_labels)
        },
        "group_strata": stratum_receipts,
        "official_test_loaded": False,
        "sealed_audit_loaded": False,
    }
    return train_rows, validation_rows, receipt


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output: {args.output_dir}")
    source_manifest = args.input_dir / "split-manifest.json"
    source_rows = read_jsonl(args.input_dir / "train.jsonl") + read_jsonl(
        args.input_dir / "validation.jsonl"
    )
    train_rows, validation_rows, receipt = build_near_duplicate_split(
        source_rows, seed=args.seed
    )
    args.output_dir.mkdir(parents=True)
    paths = {
        "train": args.output_dir / "train.jsonl",
        "validation": args.output_dir / "validation.jsonl",
    }
    for split, output_rows in (("train", train_rows), ("validation", validation_rows)):
        paths[split].write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                for row in output_rows
            ),
            encoding="utf-8",
            newline="\n",
        )
    receipt["input"] = {
        "split_manifest_sha256": sha256_file(source_manifest),
        "train_jsonl_sha256": sha256_file(args.input_dir / "train.jsonl"),
        "validation_jsonl_sha256": sha256_file(args.input_dir / "validation.jsonl"),
        "combined_rows": len(source_rows),
    }
    receipt["train_jsonl_sha256"] = sha256_file(paths["train"])
    receipt["validation_jsonl_sha256"] = sha256_file(paths["validation"])
    manifest_path = args.output_dir / "split-manifest.json"
    manifest_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    receipt["manifest_sha256"] = sha256_file(manifest_path)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
