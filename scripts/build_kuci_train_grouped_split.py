"""Create a train-only KUCI split grouped by event pair and normalized text."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

SOURCE_REVISION = "79afa1d63d1f262e6709e9be714d9ad5dae8d6f6"
SOURCE_URL = f"https://github.com/ku-nlp/KUCI/blob/{SOURCE_REVISION}/dataset/train.jsonl"
LABELS = {"a", "b", "c", "d"}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split()).casefold()


def content_key(row: dict[str, Any]) -> str:
    options = sorted(normalize_text(str(row[f"choice_{letter}"])) for letter in "abcd")
    body = "\0".join((normalize_text(str(row["context"])), *options))
    return sha256_bytes(body.encode("utf-8"))


def validate_rows(rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("training corpus is empty")
    ids: set[str] = set()
    for row in rows:
        required = {"id", "context", "core_event_pair", "label", "agreement"}
        required.update(f"choice_{letter}" for letter in "abcd")
        if required - row.keys():
            raise ValueError("KUCI row is missing required fields")
        sample_id = str(row["id"])
        if sample_id in ids:
            raise ValueError("duplicate KUCI id")
        ids.add(sample_id)
        if row["label"] not in LABELS:
            raise ValueError("KUCI label must be one of a/b/c/d")
        if not str(row["core_event_pair"]).strip():
            raise ValueError("missing core_event_pair cannot be safely grouped")
        if not normalize_text(str(row["context"])):
            raise ValueError("empty normalized context")


class _UnionFind:
    def __init__(self, count: int) -> None:
        self.parent = list(range(count))
        self.rank = [0] * count

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def grouped_split(
    rows: list[dict[str, Any]], *, seed: int = 17, validation_fraction: float = 0.10
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    validate_rows(rows)
    if not 0.05 <= validation_fraction <= 0.25:
        raise ValueError("validation_fraction must be between 0.05 and 0.25")

    union_find = _UnionFind(len(rows))
    first_by_key: dict[tuple[str, str], int] = {}
    for index, row in enumerate(rows):
        keys = (
            ("event_pair", normalize_text(str(row["core_event_pair"]))),
            ("context", normalize_text(str(row["context"]))),
            ("content", content_key(row)),
        )
        for key in keys:
            prior = first_by_key.setdefault(key, index)
            union_find.union(index, prior)

    components: dict[int, list[int]] = {}
    for index in range(len(rows)):
        components.setdefault(union_find.find(index), []).append(index)
    groups = list(components.values())
    random.Random(seed).shuffle(groups)
    target_rows = round(len(rows) * validation_fraction)
    validation_indices: set[int] = set()
    for group in groups:
        if len(validation_indices) >= target_rows:
            break
        validation_indices.update(group)

    train = [row for index, row in enumerate(rows) if index not in validation_indices]
    validation = [row for index, row in enumerate(rows) if index in validation_indices]
    if not train or not validation:
        raise ValueError("grouped split produced an empty partition")
    _assert_disjoint(train, validation)

    train_ids = {str(row["id"]) for row in train}
    validation_ids = {str(row["id"]) for row in validation}
    if train_ids & validation_ids:
        raise ValueError("sample IDs cross the split")
    train_labels = Counter(str(row["label"]) for row in train)
    validation_labels = Counter(str(row["label"]) for row in validation)
    if set(train_labels) != LABELS or set(validation_labels) != LABELS:
        raise ValueError("each split must contain all four answer positions")

    assignments = {
        str(row["id"]): "validation" if index in validation_indices else "train"
        for index, row in enumerate(rows)
    }
    manifest = {
        "source_revision": SOURCE_REVISION,
        "source_url": SOURCE_URL,
        "split_seed": seed,
        "validation_fraction_target": validation_fraction,
        "validation_fraction_realized": len(validation) / len(rows),
        "grouping_keys": ["core_event_pair", "NFKC-normalized context", "normalized option set"],
        "source_rows": len(rows),
        "connected_group_count": len(groups),
        "train_rows": len(train),
        "validation_rows": len(validation),
        "train_group_count": len(
            {
                union_find.find(index)
                for index in range(len(rows))
                if index not in validation_indices
            }
        ),
        "validation_group_count": len({union_find.find(index) for index in validation_indices}),
        "train_label_counts": dict(sorted(train_labels.items())),
        "validation_label_counts": dict(sorted(validation_labels.items())),
        "sample_assignment_sha256": sha256_bytes(
            "".join(
                f"{sample_id}\t{assignments[sample_id]}\n" for sample_id in sorted(assignments)
            ).encode("utf-8")
        ),
        "train_ids": sorted(train_ids),
        "validation_ids": sorted(validation_ids),
    }
    return train, validation, manifest


def _assert_disjoint(train: list[dict[str, Any]], validation: list[dict[str, Any]]) -> None:
    for field in ("core_event_pair",):
        left = {normalize_text(str(row[field])) for row in train}
        right = {normalize_text(str(row[field])) for row in validation}
        if left & right:
            raise ValueError(f"normalized {field} crosses the split")
    for field in ("context",):
        left = {normalize_text(str(row[field])) for row in train}
        right = {normalize_text(str(row[field])) for row in validation}
        if left & right:
            raise ValueError("normalized context crosses the split")
    if {content_key(row) for row in train} & {content_key(row) for row in validation}:
        raise ValueError("normalized question/option content crosses the split")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    source_bytes = args.source_jsonl.read_bytes()
    if (
        len(source_bytes) != 38_789_233
        or sha256_bytes(source_bytes)
        != "77cd991d35cdc5cba23d0e2208cdc66e3173577449dffdf1503bf49e0624b076"
    ):
        raise ValueError("KUCI train bytes do not match the observed pinned source")
    rows = [json.loads(line) for line in source_bytes.decode("utf-8").splitlines() if line]
    train, validation, manifest = grouped_split(
        rows, seed=args.seed, validation_fraction=args.validation_fraction
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    train_path, validation_path = (
        args.output_dir / "train.jsonl",
        args.output_dir / "validation.jsonl",
    )
    train_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in train),
        encoding="utf-8",
        newline="\n",
    )
    validation_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in validation),
        encoding="utf-8",
        newline="\n",
    )
    manifest.update(
        {
            "source_train_bytes": len(source_bytes),
            "source_train_sha256": sha256_bytes(source_bytes),
            "train_jsonl_sha256": sha256_file(train_path),
            "validation_jsonl_sha256": sha256_file(validation_path),
        }
    )
    manifest_path = args.output_dir / "split-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                key: value
                for key, value in manifest.items()
                if key not in {"train_ids", "validation_ids"}
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
