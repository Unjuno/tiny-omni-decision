"""Build a deterministic, template-grouped development split from Jamp train data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

LABELS = ("contradiction", "entailment", "neutral")
GROUP_FIELD = "template_num"
REQUIRED_FIELDS = {
    "num",
    "premise",
    "hypothesis",
    "gold_label",
    GROUP_FIELD,
}
SOURCE_REPOSITORY = "https://github.com/tomo-vv/temporalNLI_dataset"
SOURCE_REVISION = "f578b5daac7bff3855d238c032a60fd047cddd9c"
SOURCE_PATH = "dataset/train_all.tsv"


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def normalize_sentence(text: str) -> str:
    """Normalize compatibility forms and remove whitespace for overlap checks."""
    normalized = unicodedata.normalize("NFKC", text)
    return "".join(character for character in normalized if not character.isspace())


def _load_rows(source_path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with source_path.open(encoding="utf-8", newline="") as source_file:
        reader = csv.DictReader(source_file, delimiter="\t")
        fields = reader.fieldnames or []
        missing = REQUIRED_FIELDS.difference(fields)
        if missing:
            raise ValueError(f"Jamp training TSV is missing required fields: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError("Jamp training TSV contains no records")
    if any(row["gold_label"] not in LABELS for row in rows):
        raise ValueError("Jamp training TSV contains an unknown gold label")
    if len({row["num"] for row in rows}) != len(rows):
        raise ValueError("Jamp training TSV contains duplicate source record IDs")
    return rows, fields


def _group_vectors(rows: list[dict[str, str]]) -> dict[str, Counter[str]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        counts[row[GROUP_FIELD]][row["gold_label"]] += 1
    return dict(counts)


def assign_group_folds(
    rows: list[dict[str, str]], *, seed: int = 17, fold_count: int = 5
) -> dict[str, int]:
    """Greedily balance label and row counts without splitting template groups."""
    if fold_count < 2:
        raise ValueError("fold_count must be at least 2")
    vectors = _group_vectors(rows)
    if len(vectors) < fold_count:
        raise ValueError("There are fewer template groups than requested folds")

    totals = Counter(row["gold_label"] for row in rows)
    target_labels = {label: totals[label] / fold_count for label in LABELS}
    target_rows = len(rows) / fold_count
    fold_labels = [Counter() for _ in range(fold_count)]
    fold_rows = [0] * fold_count
    tie_hashes = {
        group: hashlib.sha256(f"{seed}\t{group}".encode()).hexdigest() for group in vectors
    }
    ordered_groups = sorted(
        vectors,
        key=lambda group: (-sum(vectors[group].values()), tie_hashes[group], group),
    )
    assignments: dict[str, int] = {}

    for group in ordered_groups:
        vector = vectors[group]
        candidate_scores: list[tuple[float, int, int]] = []
        for candidate_fold in range(fold_count):
            score = 0.0
            for fold in range(fold_count):
                for label in LABELS:
                    count = fold_labels[fold][label]
                    if fold == candidate_fold:
                        count += vector[label]
                    score += ((count - target_labels[label]) / target_labels[label]) ** 2
                count_rows = fold_rows[fold]
                if fold == candidate_fold:
                    count_rows += sum(vector.values())
                score += 0.25 * ((count_rows - target_rows) / target_rows) ** 2
            candidate_scores.append((score, fold_rows[candidate_fold], candidate_fold))

        selected_fold = min(candidate_scores)[2]
        assignments[group] = selected_fold
        fold_labels[selected_fold].update(vector)
        fold_rows[selected_fold] += sum(vector.values())

    return assignments


def build_split(
    rows: list[dict[str, str]], *, seed: int = 17, validation_fold: int = 0
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, int]]:
    assignments = assign_group_folds(rows, seed=seed)
    if validation_fold not in range(5):
        raise ValueError("validation_fold must be between 0 and 4")
    validation = [row for row in rows if assignments[row[GROUP_FIELD]] == validation_fold]
    train = [row for row in rows if assignments[row[GROUP_FIELD]] != validation_fold]
    if not train or not validation:
        raise ValueError("Grouped split produced an empty train or validation partition")

    train_ids = {row["num"] for row in train}
    validation_ids = {row["num"] for row in validation}
    if train_ids & validation_ids:
        raise ValueError("Source record IDs overlap between train and validation")
    train_groups = {row[GROUP_FIELD] for row in train}
    validation_groups = {row[GROUP_FIELD] for row in validation}
    if train_groups & validation_groups:
        raise ValueError("Template groups overlap between train and validation")

    seen_pairs: dict[tuple[str, str], str] = {}
    for row in rows:
        pair = (
            normalize_sentence(row["premise"]),
            normalize_sentence(row["hypothesis"]),
        )
        if pair in seen_pairs:
            raise ValueError(
                "Normalized premise/hypothesis pair repeats in source rows "
                f"{seen_pairs[pair]} and {row['num']}"
            )
        seen_pairs[pair] = row["num"]

    return train, validation, assignments


def _serialize_rows(rows: list[dict[str, str]], fields: list[str]) -> bytes:
    from io import StringIO

    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _counts(rows: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "unique_source_ids": len({row["num"] for row in rows}),
        "unique_template_groups": len({row[GROUP_FIELD] for row in rows}),
        "labels": dict(sorted(Counter(row["gold_label"] for row in rows).items())),
        "template_groups": sorted({row[GROUP_FIELD] for row in rows}, key=int),
    }


def create_frozen_split(source_path: Path, output_dir: Path, *, seed: int = 17) -> dict[str, Any]:
    source_bytes = source_path.read_bytes()
    rows, fields = _load_rows(source_path)
    train, validation, assignments = build_split(rows, seed=seed)
    train_bytes = _serialize_rows(train, fields)
    validation_bytes = _serialize_rows(validation, fields)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "train.tsv").write_bytes(train_bytes)
    (output_dir / "validation.tsv").write_bytes(validation_bytes)

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "dataset": "Jamp controlled Japanese temporal inference",
        "source_repository": SOURCE_REPOSITORY,
        "source_revision": SOURCE_REVISION,
        "source_path": SOURCE_PATH,
        "source_declared_license": "CC-BY-SA-4.0",
        "source_license_note": (
            "The paper states JSeM is BSD-3-Clause and that use of JSeM and the "
            "GSK-distributed case-frame dictionary follows their license terms. "
            "Only Jamp's published train_all.tsv was used; the case-frame dictionary "
            "itself was not acquired. Preserve Jamp attribution and CC BY-SA obligations."
        ),
        "citation": (
            "Sugimoto, Onoe, and Yanaka (2023), Jamp: Controlled Japanese Temporal "
            "Inference Dataset for Evaluating Generalization Capacity of Language Models, "
            "ACL SRW, https://aclanthology.org/2023.acl-srw.8/"
        ),
        "source_file_sha256": sha256_bytes(source_bytes),
        "split": {
            "algorithm": "deterministic greedy five-fold label-and-row balance over template_num",
            "seed": seed,
            "validation_fold": 0,
            "fold_count": 5,
            "group_field": GROUP_FIELD,
            "fold_by_template": dict(sorted(assignments.items(), key=lambda item: int(item[0]))),
            "train": _counts(train),
            "validation": _counts(validation),
            "train_validation_group_overlap": 0,
            "train_validation_id_overlap": 0,
            "normalized_pair_overlap": 0,
        },
        "train_sha256": sha256_bytes(train_bytes),
        "validation_sha256": sha256_bytes(validation_bytes),
    }
    (output_dir / "split-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    manifest = create_frozen_split(args.source, args.output_dir, seed=args.seed)
    print(json.dumps(manifest["split"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
