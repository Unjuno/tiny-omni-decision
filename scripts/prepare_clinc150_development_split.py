"""Build a deterministic exact-text-grouped CLINC150 train/validation split."""

from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

DATASET_ID = "clinc/clinc_oos"
DATASET_REVISION = "155b9c710419136e17307b80d0a13e68cd46b4ec"
CONFIG_NAME = "plus"
README_SHA256 = "e28704a3c04f6b05c286ca2a8891bbf32c69c000dedfc5825d6e1713369b7a89"
SOURCE_FILE_HASHES = {
    "train": "30188119cf9f86fc9db27e1c22442d091cb5cb0913c9496f945fe11e7a02a28f",
    "validation": "fbd545b46c611c4a7ba4b48cae6c7f09bb5b59f33ff56206ad1cd366c85cdfaa",
}
PROTOCOL_ID = "clinc150-content-grouped-development-v1"
SEED = 17
VALIDATION_FRACTION = 0.20


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_text(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("CLINC utterances must be nonempty strings")
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def build_grouped_split(
    rows: list[dict[str, Any]],
    class_names: list[str],
    *,
    seed: int = SEED,
    validation_fraction: float = VALIDATION_FRACTION,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Assign every exact-normalized-content group wholly to one split."""
    if len(class_names) != 151 or len(set(class_names)) != len(class_names):
        raise ValueError("CLINC150 plus must have 151 unique class names")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must lie strictly between zero and one")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(rows):
        text = normalize_text(row["text"])
        label = int(row["intent"])
        if label < 0 or label >= len(class_names):
            raise ValueError(f"unexpected label index {label}")
        content_group_id = sha256_bytes(text.encode("utf-8"))
        grouped[content_group_id].append(
            {
                "source_row": index,
                "text": row["text"],
                "label": label,
                "content_group_id": content_group_id,
            }
        )

    strata: dict[tuple[int, ...], list[str]] = defaultdict(list)
    conflicting_group_count = 0
    for group_id, group_rows in grouped.items():
        labels = tuple(sorted({row["label"] for row in group_rows}))
        if len(labels) > 1:
            conflicting_group_count += 1
        strata[labels].append(group_id)

    validation_group_ids: set[str] = set()
    stratum_receipts: dict[str, dict[str, int]] = {}
    for labels, group_ids in sorted(strata.items()):
        ordered = sorted(
            group_ids,
            key=lambda group_id: hashlib.sha256(
                f"{seed}:{PROTOCOL_ID}:{','.join(map(str, labels))}:{group_id}".encode()
            ).hexdigest(),
        )
        validation_count = int(round(len(ordered) * validation_fraction))
        if len(ordered) >= 2:
            validation_count = min(max(validation_count, 1), len(ordered) - 1)
        elif validation_count:
            validation_count = 1
        validation_group_ids.update(ordered[:validation_count])
        stratum_receipts[",".join(map(str, labels))] = {
            "content_groups": len(ordered),
            "validation_content_groups": validation_count,
        }

    train_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    train_labels: Counter[int] = Counter()
    validation_labels: Counter[int] = Counter()
    for group_id, group_rows in grouped.items():
        destination = validation_rows if group_id in validation_group_ids else train_rows
        label_counts = validation_labels if group_id in validation_group_ids else train_labels
        for row in group_rows:
            record = {
                "sample_id": sha256_bytes(
                    f"{row['source_row']}:{row['content_group_id']}:{row['label']}".encode()
                ),
                "content_group_id": row["content_group_id"],
                "text": row["text"],
                "label": row["label"],
                "label_name": class_names[row["label"]],
            }
            destination.append(record)
            label_counts[row["label"]] += 1

    train_rows.sort(key=lambda row: row["sample_id"])
    validation_rows.sort(key=lambda row: row["sample_id"])
    train_groups = {row["content_group_id"] for row in train_rows}
    validation_groups = {row["content_group_id"] for row in validation_rows}
    if train_groups & validation_groups:
        raise AssertionError("content group leakage after deterministic assignment")
    if len(train_rows) + len(validation_rows) != len(rows):
        raise AssertionError("split failed to preserve every source row")
    missing_train = set(range(len(class_names))) - set(train_labels)
    missing_validation = set(range(len(class_names))) - set(validation_labels)
    if missing_train or missing_validation:
        raise ValueError(
            f"class coverage missing: train={sorted(missing_train)}, "
            f"validation={sorted(missing_validation)}"
        )

    receipt = {
        "protocol_id": PROTOCOL_ID,
        "seed": seed,
        "validation_fraction_target": validation_fraction,
        "group_rule": "SHA-256(NFKC(casefold(text)), collapsed whitespace)",
        "assignment_rule": (
            "within each sorted tuple of labels per content group, order by "
            "SHA-256(seed:protocol_id:label-tuple:content_group_id) and allocate "
            "the rounded validation fraction, preserving groups intact"
        ),
        "source_rows": len(rows),
        "source_unique_normalized_content_groups": len(grouped),
        "conflicting_label_content_groups": conflicting_group_count,
        "train_rows": len(train_rows),
        "train_content_groups": len(train_groups),
        "validation_rows": len(validation_rows),
        "validation_content_groups": len(validation_groups),
        "cross_split_content_group_overlap": len(train_groups & validation_groups),
        "train_class_counts": {str(key): train_labels[key] for key in range(len(class_names))},
        "validation_class_counts": {
            str(key): validation_labels[key] for key in range(len(class_names))
        },
        "group_strata": stratum_receipts,
        "official_test_loaded": False,
    }
    return train_rows, validation_rows, receipt


def read_class_names(readme_path: Path) -> list[str]:
    if sha256_file(readme_path) != README_SHA256:
        raise ValueError("Pinned CLINC150 README hash mismatch")
    text = readme_path.read_text(encoding="utf-8")
    parts = text.split("---", 2)
    if len(parts) != 3:
        raise ValueError("Pinned dataset card has no valid YAML front matter")
    card = yaml.safe_load(parts[1])
    plus = next(
        (entry for entry in card["dataset_info"] if entry["config_name"] == CONFIG_NAME), None
    )
    if plus is None:
        raise ValueError("Pinned dataset card has no plus config")
    feature = next(item for item in plus["features"] if item["name"] == "intent")
    names_by_id = feature["dtype"]["class_label"]["names"]
    names = [names_by_id[str(index)] for index in range(151)]
    expected_counts = {split["name"]: split["num_examples"] for split in plus["splits"]}
    if expected_counts != {"train": 15250, "validation": 3100, "test": 5500}:
        raise ValueError(f"Unexpected CLINC150 plus split counts: {expected_counts}")
    return names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite split output: {args.output_dir}")
    source_dir = args.source_dir
    readme_path = source_dir / "README.md"
    class_names = read_class_names(readme_path)
    import pyarrow.parquet as parquet

    all_rows: list[dict[str, Any]] = []
    file_receipts: dict[str, dict[str, Any]] = {}
    for split in ("train", "validation"):
        relative = f"plus/{split}-00000-of-00001.parquet"
        path = source_dir / relative
        actual_hash = sha256_file(path)
        if actual_hash != SOURCE_FILE_HASHES[split]:
            raise ValueError(f"Pinned CLINC150 {split} hash mismatch")
        rows = parquet.read_table(path, columns=["text", "intent"]).to_pylist()
        if len(rows) != (15250 if split == "train" else 3100):
            raise ValueError(f"Unexpected {split} row count: {len(rows)}")
        offset = len(all_rows)
        all_rows.extend(rows)
        file_receipts[split] = {
            "path": relative,
            "sha256": actual_hash,
            "bytes": path.stat().st_size,
            "row_count": len(rows),
            "source_index_range_start": offset,
        }

    train_rows, validation_rows, split_receipt = build_grouped_split(all_rows, class_names)
    args.output_dir.mkdir(parents=True)
    paths = {
        "train": args.output_dir / "train.jsonl",
        "validation": args.output_dir / "validation.jsonl",
    }
    for split, rows in (("train", train_rows), ("validation", validation_rows)):
        paths[split].write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows
            ),
            encoding="utf-8",
        )
    split_receipt["source"] = {
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "config": CONFIG_NAME,
        "source_files": file_receipts,
        "license": "CC-BY-3.0",
        "upstream_repo": "https://github.com/clinc/oos-eval",
        "upstream_revision": "828f8093932c8fe6ca7936c3d2e52903b1c523de",
    }
    split_receipt["class_names"] = class_names
    split_receipt["train_jsonl_sha256"] = sha256_file(paths["train"])
    split_receipt["validation_jsonl_sha256"] = sha256_file(paths["validation"])
    split_receipt["official_test_downloaded_or_opened"] = False
    manifest_path = args.output_dir / "split-manifest.json"
    manifest_path.write_text(
        json.dumps(split_receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    split_receipt["split_manifest_sha256"] = sha256_file(manifest_path)
    print(json.dumps(split_receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
