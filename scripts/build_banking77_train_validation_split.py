"""Build a content-grouped development split from pinned BANKING77 TRAIN only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROTOCOL_ID = "banking77-compact-bert-decision-v1"
EXPECTED_REVISION = "1846984189f2bc0ac51c6fc3a600fa9e1714e0c6"
EXPECTED_TRAIN_SHA256 = "8dc89b87a188509c42cce5a086e8f7c1ff4a225bdd9ae3f23e6cdfa5806e363a"


def normalize_utterance(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build_split_manifest(
    rows: list[dict[str, Any]],
    *,
    source_revision: str,
    source_sha256: str,
    validation_fraction: float = 0.2,
    seed: int = 17,
) -> dict[str, Any]:
    """Make a deterministic class-stratified split, keeping duplicate text together."""
    if source_revision != EXPECTED_REVISION:
        raise ValueError("Unexpected BANKING77 revision")
    if source_sha256.lower() != EXPECTED_TRAIN_SHA256:
        raise ValueError("Unexpected BANKING77 TRAIN hash")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be strictly between zero and one")
    if not rows:
        raise ValueError("TRAIN rows are empty")

    group_rows: dict[str, list[int]] = defaultdict(list)
    group_labels: dict[str, str] = {}
    for index, row in enumerate(rows):
        text = row.get("text")
        label = row.get("label")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Row {index} has no non-empty text")
        if label is None:
            raise ValueError(f"Row {index} has no label")
        group_id = sha256_text(normalize_utterance(text))
        label_id = str(label)
        previous_label = group_labels.setdefault(group_id, label_id)
        if previous_label != label_id:
            raise ValueError("Normalized duplicate utterance has conflicting labels")
        group_rows[group_id].append(index)

    groups_by_label: dict[str, list[str]] = defaultdict(list)
    for group_id, label in group_labels.items():
        groups_by_label[label].append(group_id)

    validation_groups: set[str] = set()
    for label, group_ids in groups_by_label.items():
        if len(group_ids) < 2:
            raise ValueError(f"Class {label} has fewer than two unique utterances")
        ordered = sorted(
            group_ids,
            key=lambda group_id: sha256_text(
                f"{seed}:{PROTOCOL_ID}:{label}:{group_id}"
            ),
        )
        validation_count = max(
            1, min(len(ordered) - 1, math.floor(len(ordered) * validation_fraction + 0.5))
        )
        validation_groups.update(ordered[:validation_count])

    validation_indices = [
        index
        for index in range(len(rows))
        if sha256_text(normalize_utterance(rows[index]["text"])) in validation_groups
    ]
    train_indices = sorted(set(range(len(rows))) - set(validation_indices))
    validation_indices = sorted(validation_indices)
    train_index_set = set(train_indices)
    train_groups = {
        group_id for group_id, indices in group_rows.items()
        if any(index in train_index_set for index in indices)
    }
    if train_groups & validation_groups:
        raise AssertionError("Content group overlap after split")
    if set(train_indices) & set(validation_indices):
        raise AssertionError("Row overlap after split")
    if len(train_indices) + len(validation_indices) != len(rows):
        raise AssertionError("Split does not cover all TRAIN rows")

    all_labels = sorted({str(row["label"]) for row in rows})
    train_counts = Counter(str(rows[index]["label"]) for index in train_indices)
    validation_counts = Counter(str(rows[index]["label"]) for index in validation_indices)
    if set(train_counts) != set(all_labels) or set(validation_counts) != set(all_labels):
        raise ValueError("Every class must occur in both train and validation")

    row_ids = [sha256_text(f"{source_sha256.lower()}:{index}") for index in range(len(rows))]

    def ordered_hash(values: list[str]) -> str:
        return sha256_text("\n".join(values))

    return {
        "format": "banking77_content_grouped_train_validation_v1",
        "protocol_id": PROTOCOL_ID,
        "source_revision": source_revision,
        "source_train_sha256": source_sha256.lower(),
        "seed": seed,
        "validation_fraction_target": validation_fraction,
        "group_rule": "sha256(NFKC(casefold(text)), whitespace collapsed)",
        "assignment_rule": (
            "within each label, sort unique groups by "
            "SHA256(seed:protocol_id:label:group_sha256); first rounded fraction "
            "goes to validation"
        ),
        "row_count": len(rows),
        "class_count": len(all_labels),
        "unique_content_group_count": len(group_rows),
        "train": {
            "row_indices": train_indices,
            "row_count": len(train_indices),
            "unique_content_groups": len(train_groups),
            "ordered_row_ids_sha256": ordered_hash([row_ids[i] for i in train_indices]),
            "group_ids_sha256": ordered_hash(sorted(train_groups)),
            "class_counts": {label: train_counts[label] for label in all_labels},
        },
        "validation": {
            "row_indices": validation_indices,
            "row_count": len(validation_indices),
            "unique_content_groups": len(validation_groups),
            "ordered_row_ids_sha256": ordered_hash([row_ids[i] for i in validation_indices]),
            "group_ids_sha256": ordered_hash(sorted(validation_groups)),
            "class_counts": {label: validation_counts[label] for label in all_labels},
        },
        "overlap": {
            "row_indices": 0,
            "normalized_content_groups": len(train_groups & validation_groups),
        },
        "official_test_loaded": False,
        "independent_source_identity_verified": False,
    }


def load_pinned_train(path: Path) -> list[dict[str, Any]]:
    """Read only the pinned TRAIN parquet; the official test path is never accepted."""
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - exercised in runtime environments
        raise RuntimeError("Install the project's data-analysis extras with pyarrow") from exc
    frame = pd.read_parquet(path)
    if list(frame.columns) != ["text", "label"]:
        raise ValueError("Unexpected BANKING77 TRAIN schema")
    return frame.to_dict(orient="records")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-parquet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    actual_hash = hashlib.sha256(args.train_parquet.read_bytes()).hexdigest()
    rows = load_pinned_train(args.train_parquet)
    manifest = build_split_manifest(
        rows,
        source_revision=EXPECTED_REVISION,
        source_sha256=actual_hash,
    )
    args.output.mkdir(parents=True)
    (args.output / "split-manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    for split in ("train", "validation"):
        index_key = "row_indices"
        rows_out = []
        for source_index in manifest[split][index_key]:
            row = rows[source_index]
            group_id = sha256_text(normalize_utterance(row["text"]))
            rows_out.append(
                {
                    "sample_id": sha256_text(f"{actual_hash}:{source_index}"),
                    "source_row_index": source_index,
                    "content_group_id": group_id,
                    "text": row["text"],
                    "label": int(row["label"]),
                }
            )
        with (args.output / f"{split}.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
            for row in rows_out:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {k: v for k, v in manifest.items() if k not in {"train", "validation"}}
    print(json.dumps(summary, indent=2))
    train_count = manifest["train"]["row_count"]
    validation_count = manifest["validation"]["row_count"]
    print(f"train={train_count} validation={validation_count}")


if __name__ == "__main__":
    main()
