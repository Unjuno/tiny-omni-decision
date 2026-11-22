"""Prepare JCommonsenseQA v1.1 with exact-question groups kept in one split."""

from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SOURCE_COMMIT = "6791a5003e6deaa6fb2458bcb3aa1f58dc52ef33"
SOURCE_TAG_OBJECT = "6dbf79c766d3e64a1a575ef474cb654b6c9385d4"
SOURCE_HASHES = {
    "train": "641c254ba52924fe20837d5694a50d35da2312b3dcf4983d2b00736976cbe6c7",
    "validation": "01c73e37420368da82729ee82cc6279f5d9b6a22b2e2fb8dae814261fae13bea",
    "README.md": "a92dca61bb91c38eed4ec49b091e258a4e5f28c225d18741dedcc27e2e87d2eb",
    "LICENSE": "b41b90bc49eef8ca3387304e839ae8059b917e80346028493bccc0597a47fbbd",
}
PROTOCOL = "jcommonsenseqa-question-grouped-development-v1"
SEED = 17
VALIDATION_GROUP_FRACTION = 0.20
CHOICE_KEYS = tuple(f"choice{index}" for index in range(5))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_text(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("questions and choices must be nonempty strings")
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def target_text_is_ambiguous(row: dict[str, Any]) -> bool:
    target = normalize_text(row[CHOICE_KEYS[int(row["label"])]])
    return sum(normalize_text(row[key]) == target for key in CHOICE_KEYS) > 1


def read_json_lines(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        row["source_row"] = line_number - 1
        rows.append(row)
    return rows


def validate_rows(rows: list[dict[str, Any]]) -> None:
    seen_ids: set[int] = set()
    for row in rows:
        q_id = int(row["q_id"])
        if q_id in seen_ids:
            raise ValueError(f"duplicate q_id: {q_id}")
        seen_ids.add(q_id)
        normalize_text(row["question"])
        for key in CHOICE_KEYS:
            normalize_text(row[key])
        label = int(row["label"])
        if label not in range(5):
            raise ValueError(f"question {q_id} has invalid target label {label}")


def build_grouped_split(
    rows: list[dict[str, Any]],
    *,
    seed: int = SEED,
    validation_group_fraction: float = VALIDATION_GROUP_FRACTION,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Reassign every row by normalized question identity; do not drop repeats."""
    if not rows:
        raise ValueError("cannot split an empty dataset")
    if not 0 < validation_group_fraction < 1:
        raise ValueError("validation_group_fraction must be between 0 and 1")
    validate_rows(rows)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        question_key = normalize_text(row["question"])
        group_id = sha256_text(question_key)
        grouped[group_id].append(row)

    ordered_groups = sorted(
        grouped,
        key=lambda group_id: hashlib.sha256(f"{seed}:{PROTOCOL}:{group_id}".encode()).hexdigest(),
    )
    validation_group_count = round(len(ordered_groups) * validation_group_fraction)
    if len(ordered_groups) >= 2:
        validation_group_count = min(max(validation_group_count, 1), len(ordered_groups) - 1)
    validation_ids = set(ordered_groups[:validation_group_count])

    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    for group_id, group_rows in grouped.items():
        destination = validation if group_id in validation_ids else train
        for row in group_rows:
            clean = {key: value for key, value in row.items() if key != "source_row"}
            clean["sample_id"] = f"jcommonsenseqa-{int(row['q_id'])}"
            clean["content_group_id"] = group_id
            clean["source_split"] = str(row["source_split"])
            clean["label"] = int(row["label"])
            clean["q_id"] = int(row["q_id"])
            destination.append(clean)
    train.sort(key=lambda row: row["sample_id"])
    validation.sort(key=lambda row: row["sample_id"])
    train_groups = {row["content_group_id"] for row in train}
    validation_groups = {row["content_group_id"] for row in validation}
    if train_groups & validation_groups:
        raise AssertionError("normalized question group leaked across splits")
    if len(train) + len(validation) != len(rows):
        raise AssertionError("split did not preserve all source rows")
    question_groups = list(grouped.values())
    repeated_groups = [group for group in question_groups if len(group) > 1]
    conflicting_answer_groups = 0
    for group in repeated_groups:
        answers = {normalize_text(row[CHOICE_KEYS[int(row["label"])]]) for row in group}
        conflicting_answer_groups += len(answers) > 1
    duplicate_choice_rows = 0
    ambiguous_target_rows = 0
    for row in rows:
        options = [normalize_text(row[key]) for key in CHOICE_KEYS]
        duplicate_choice_rows += len(set(options)) != len(options)
        ambiguous_target_rows += target_text_is_ambiguous(row)
    receipt = {
        "protocol": PROTOCOL,
        "seed": seed,
        "group_rule": "SHA-256(NFKC(casefold(question)), collapsed whitespace)",
        "assignment_rule": (
            "sort content-group SHA-256 values by SHA-256(seed:protocol:group_id); "
            "assign the first rounded 20% of groups to validation"
        ),
        "source_rows": len(rows),
        "unique_question_groups": len(grouped),
        "repeated_question_groups": len(repeated_groups),
        "conflicting_correct_answer_groups_retained": conflicting_answer_groups,
        "rows_with_duplicate_choice_text_retained": duplicate_choice_rows,
        "rows_with_ambiguous_duplicate_target_retained": ambiguous_target_rows,
        "train_rows": len(train),
        "train_question_groups": len(train_groups),
        "validation_rows": len(validation),
        "validation_question_groups": len(validation_groups),
        "cross_split_question_group_overlap": len(train_groups & validation_groups),
        "train_label_counts": {
            str(label): Counter(row["label"] for row in train)[label] for label in range(5)
        },
        "validation_label_counts": {
            str(label): Counter(row["label"] for row in validation)[label] for label in range(5)
        },
        "official_test_downloaded_or_opened": False,
    }
    return train, validation, receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_dir}")
    source_dir = args.source_dir.resolve()
    split_paths = {
        "train": source_dir / "datasets/jcommonsenseqa-v1.1/train-v1.1.json",
        "validation": source_dir / "datasets/jcommonsenseqa-v1.1/valid-v1.1.json",
    }
    hashes = {name: sha256_file(path) for name, path in split_paths.items()}
    if hashes != {key: SOURCE_HASHES[key] for key in ("train", "validation")}:
        raise ValueError("pinned JCommonsenseQA source data hash mismatch")
    for name in ("README.md", "LICENSE"):
        if sha256_file(source_dir / name) != SOURCE_HASHES[name]:
            raise ValueError(f"pinned JGLUE {name} hash mismatch")

    rows: list[dict[str, Any]] = []
    source_counts = {}
    for split, path in split_paths.items():
        split_rows = read_json_lines(path)
        expected_count = 8939 if split == "train" else 1119
        if len(split_rows) != expected_count:
            raise ValueError(f"unexpected {split} row count: {len(split_rows)}")
        for row in split_rows:
            row["source_split"] = split
        validate_rows(split_rows)
        if {int(row["label"]) for row in split_rows} != set(range(5)):
            raise ValueError(f"official {split} is missing one or more choice-position labels")
        source_counts[split] = len(split_rows)
        rows.extend(split_rows)
    train, validation, receipt = build_grouped_split(rows)
    for split_name, split_rows in (("train", train), ("validation", validation)):
        if {int(row["label"]) for row in split_rows} != set(range(5)):
            raise ValueError(f"grouped {split_name} is missing a choice-position label")
    args.output_dir.mkdir(parents=True)
    output_paths = {
        "train": args.output_dir / "train.jsonl",
        "validation": args.output_dir / "validation.jsonl",
    }
    for name, split_rows in (("train", train), ("validation", validation)):
        output_paths[name].write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                for row in split_rows
            ),
            encoding="utf-8",
        )
    receipt["source"] = {
        "repository": "https://github.com/yahoojapan/JGLUE",
        "tag": "v1.1.0",
        "tag_object": SOURCE_TAG_OBJECT,
        "commit": SOURCE_COMMIT,
        "license": "CC-BY-SA-4.0",
        "source_split_row_counts": source_counts,
        "source_files_sha256": hashes,
        "official_test_file_present_in_release": False,
        "official_test_downloaded_or_opened": False,
    }
    receipt["class_names"] = ["choice0", "choice1", "choice2", "choice3", "choice4"]
    receipt["train_jsonl_sha256"] = sha256_file(output_paths["train"])
    receipt["validation_jsonl_sha256"] = sha256_file(output_paths["validation"])
    (args.output_dir / "split-manifest.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
