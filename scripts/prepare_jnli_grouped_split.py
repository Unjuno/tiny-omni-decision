"""Freeze a group-disjoint train/validation split from JGLUE JNLI train and dev only."""

from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SOURCE_COMMIT = "6791a5003e6deaa6fb2458bcb3aa1f58dc52ef33"
SOURCE_HASHES = {
    "train-v1.1.json": "4367d2969587047e2de70178a67baab964ac1dcb7e043ca719a9468d2662bafe",
    "valid-v1.1.json": "1d00fbfa77abd4ec538e986ab840f99d5d63c79b0f37b4b6a225350750190e80",
}
LABELS = ("entailment", "contradiction", "neutral")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def parse_image_ids(yjcaptions_id: str) -> tuple[str, ...]:
    image_part, *sentence_ids = yjcaptions_id.rsplit("-", 2)
    image_ids = tuple(image_part.split("_"))
    if len(sentence_ids) != 2 or len(image_ids) not in (1, 2):
        raise ValueError(f"malformed yjcaptions_id: {yjcaptions_id!r}")
    if any(not image_id.isdigit() for image_id in image_ids):
        raise ValueError(f"non-numeric image ID in yjcaptions_id: {yjcaptions_id!r}")
    return image_ids


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.rank: dict[str, int] = {}

    def find(self, value: str) -> str:
        if value not in self.parent:
            self.parent[value] = value
            self.rank[value] = 0
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def _read_jsonl(path: Path, split_name: str) -> list[dict[str, Any]]:
    expected_name = "train-v1.1.json" if split_name == "publisher_train" else "valid-v1.1.json"
    if path.name != expected_name:
        raise ValueError(f"expected {expected_name} for {split_name}, got {path.name}")
    if sha256_file(path) != SOURCE_HASHES[expected_name]:
        raise ValueError(f"source hash mismatch for {path}")
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        row = json.loads(line)
        if set(row) != {"sentence_pair_id", "yjcaptions_id", "sentence1", "sentence2", "label"}:
            raise ValueError(f"unexpected JNLI schema at {path.name}:{line_number}")
        if row["label"] not in LABELS:
            raise ValueError(f"unknown JNLI label at {path.name}:{line_number}")
        if not row["sentence1"].strip() or not row["sentence2"].strip():
            raise ValueError(f"empty sentence at {path.name}:{line_number}")
        row = dict(row)
        row["source_split"] = split_name
        row["source_row"] = line_number
        rows.append(row)
    if not rows:
        raise ValueError(f"empty source file: {path}")
    return rows


def build_grouped_split(
    publisher_train: list[dict[str, Any]],
    publisher_dev: list[dict[str, Any]],
    *,
    seed: int = 17,
    validation_fraction: float = 0.2,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be in (0, 1)")
    rows = publisher_train + publisher_dev
    union_find = _UnionFind()
    row_entities: list[tuple[str, ...]] = []
    for row in rows:
        entities = [f"image:{image_id}" for image_id in parse_image_ids(row["yjcaptions_id"])]
        entities.extend(
            (
                f"sentence:{normalize_text(row['sentence1'])}",
                f"sentence:{normalize_text(row['sentence2'])}",
            )
        )
        for entity in entities[1:]:
            union_find.union(entities[0], entity)
        row_entities.append(tuple(entities))

    component_entities: dict[str, set[str]] = defaultdict(set)
    for entities in row_entities:
        for entity in entities:
            component_entities[union_find.find(entity)].add(entity)
    component_ids = {
        root: hashlib.sha256("\n".join(sorted(entities)).encode("utf-8")).hexdigest()
        for root, entities in component_entities.items()
    }
    for row, entities in zip(rows, row_entities, strict=True):
        row["content_group_id"] = component_ids[union_find.find(entities[0])]
        row["image_ids"] = list(parse_image_ids(row["yjcaptions_id"]))
        row["sample_id"] = f"{row['source_split']}:{row['source_row']}:{row['sentence_pair_id']}"

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["content_group_id"]].append(row)
    target_rows = round(len(rows) * validation_fraction)
    group_order = sorted(
        groups,
        key=lambda group: hashlib.sha256(
            f"jnli-group-split-v1|seed={seed}|{group}".encode()
        ).hexdigest(),
    )
    validation_groups: set[str] = set()
    validation_rows = 0
    for group in group_order:
        if validation_rows >= target_rows:
            break
        validation_groups.add(group)
        validation_rows += len(groups[group])

    train = [row for row in rows if row["content_group_id"] not in validation_groups]
    validation = [row for row in rows if row["content_group_id"] in validation_groups]
    train_groups = {row["content_group_id"] for row in train}
    validation_group_ids = {row["content_group_id"] for row in validation}
    if train_groups & validation_group_ids:
        raise AssertionError("group overlap in generated split")

    train_images = {image for row in train for image in row["image_ids"]}
    validation_images = {image for row in validation for image in row["image_ids"]}
    train_sentences = {
        normalize_text(row[field]) for row in train for field in ("sentence1", "sentence2")
    }
    validation_sentences = {
        normalize_text(row[field]) for row in validation for field in ("sentence1", "sentence2")
    }
    if train_images & validation_images or train_sentences & validation_sentences:
        raise AssertionError("underlying image or normalized sentence overlap")

    def count(rows_: list[dict[str, Any]], key: str) -> dict[str, int]:
        return dict(sorted(Counter(str(row[key]) for row in rows_).items()))

    all_group_sizes = Counter(row["content_group_id"] for row in rows)
    manifest = {
        "schema_version": 1,
        "protocol": "jnli-yjcaptions-image-and-normalized-sentence-grouped-v1",
        "source": {
            "repository": "https://github.com/yahoojapan/JGLUE",
            "revision": SOURCE_COMMIT,
            "license": "CC-BY-SA-4.0",
        },
        "seed": seed,
        "validation_fraction_target": validation_fraction,
        "rows": len(rows),
        "train_rows": len(train),
        "validation_rows": len(validation),
        "groups": len(groups),
        "train_groups": len(train_groups),
        "validation_groups": len(validation_group_ids),
        "maximum_group_size": max(all_group_sizes.values()),
        "group_overlap": 0,
        "underlying_image_overlap": 0,
        "normalized_sentence_overlap": 0,
        "train_labels": count(train, "label"),
        "validation_labels": count(validation, "label"),
        "train_source_rows": count(train, "source_split"),
        "validation_source_rows": count(validation, "source_split"),
        "official_test_downloaded_or_opened": False,
    }
    return train, validation, manifest


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publisher-train", type=Path, required=True)
    parser.add_argument("--publisher-dev", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    source_train = _read_jsonl(args.publisher_train, "publisher_train")
    source_dev = _read_jsonl(args.publisher_dev, "publisher_dev")
    train, validation, manifest = build_grouped_split(
        source_train,
        source_dev,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
    )
    manifest["source"]["publisher_train_sha256"] = sha256_file(args.publisher_train)
    manifest["source"]["publisher_dev_sha256"] = sha256_file(args.publisher_dev)
    args.output_dir.mkdir(parents=True)
    _write_jsonl(args.output_dir / "train.jsonl", train)
    _write_jsonl(args.output_dir / "validation.jsonl", validation)
    manifest["train_jsonl_sha256"] = sha256_file(args.output_dir / "train.jsonl")
    manifest["validation_jsonl_sha256"] = sha256_file(args.output_dir / "validation.jsonl")
    (args.output_dir / "split-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest["split_manifest_sha256"] = sha256_file(args.output_dir / "split-manifest.json")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
