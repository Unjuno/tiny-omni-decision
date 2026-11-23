"""Validate and deterministically group-split licensed JAQKET Dev1 rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _normalized(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    return re.sub(r"\s+", " ", text).strip()


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not rows:
        raise ValueError("Dev1 input is empty")
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"row {index} is not an object")
        for field in ("qid", "question", "answer_entity", "answer_candidates"):
            if field not in row:
                raise ValueError(f"row {index} is missing {field}")
        if not str(row["qid"]).strip() or not str(row["question"]).strip():
            raise ValueError(f"row {index} has an empty qid or question")
        options = row["answer_candidates"]
        if not isinstance(options, list) or len(options) != 20:
            raise ValueError(f"row {index} must contain exactly 20 answer candidates")
        normalized_options = [_normalized(option) for option in options]
        if any(not option for option in normalized_options):
            raise ValueError(f"row {index} contains an empty candidate")
        if len(set(normalized_options)) != 20:
            raise ValueError(f"row {index} contains duplicate normalized candidates")
        if normalized_options.count(_normalized(row["answer_entity"])) != 1:
            raise ValueError(f"row {index} target must occur exactly once among candidates")
    qids = [str(row["qid"]) for row in rows]
    if len(set(qids)) != len(qids):
        raise ValueError("qid values must be unique")
    return rows


def _grouped_split(
    rows: list[dict[str, Any]], seed: int, validation_fraction: float
) -> tuple[list[int], list[int], int]:
    parents = list(range(len(rows)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    # Question, answer identity and exact unordered option-set identity are
    # conservatively kept together across train and validation.
    key_functions = (
        lambda row: _normalized(row["question"]),
        lambda row: _normalized(row["answer_entity"]),
        lambda row: tuple(sorted(_normalized(option) for option in row["answer_candidates"])),
    )
    for key_function in key_functions:
        first_seen: dict[Any, int] = {}
        for index, row in enumerate(rows):
            key = key_function(row)
            if key in first_seen:
                union(index, first_seen[key])
            else:
                first_seen[key] = index

    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(rows)):
        groups[find(index)].append(index)
    ranked_groups = []
    prefix = f"seed{seed}:jaqket-dev1-answer-group-v1:"
    for indices in groups.values():
        group_id = min(str(rows[index]["qid"]) for index in indices)
        rank = hashlib.sha256((prefix + group_id).encode("utf-8")).hexdigest()
        ranked_groups.append((rank, indices))
    ranked_groups.sort(key=lambda item: item[0])

    target_rows = round(validation_fraction * len(rows))
    validation: set[int] = set()
    for _, indices in ranked_groups:
        if len(validation) < target_rows and abs(
            len(validation) + len(indices) - target_rows
        ) <= abs(len(validation) - target_rows):
            validation.update(indices)
    train = set(range(len(rows))) - validation
    return sorted(train), sorted(validation), len(groups)


def create_split(input_path: Path, output_dir: Path, *, seed: int = 17) -> dict[str, Any]:
    source_bytes = input_path.read_bytes()
    rows = _read_rows(input_path)
    train_indices, validation_indices, group_components = _grouped_split(rows, seed, 0.2)
    train_rows = [rows[index] for index in train_indices]
    validation_rows = [rows[index] for index in validation_indices]

    def identities(split_rows: list[dict[str, Any]], field: str) -> set[str]:
        return {_normalized(row[field]) for row in split_rows}

    train_questions, validation_questions = (
        identities(train_rows, "question"),
        identities(validation_rows, "question"),
    )
    train_targets, validation_targets = (
        identities(train_rows, "answer_entity"),
        identities(validation_rows, "answer_entity"),
    )
    train_candidate_sets = {
        tuple(sorted(_normalized(option) for option in row["answer_candidates"]))
        for row in train_rows
    }
    validation_candidate_sets = {
        tuple(sorted(_normalized(option) for option in row["answer_candidates"]))
        for row in validation_rows
    }
    train_qids = {str(row["qid"]) for row in train_rows}
    validation_qids = {str(row["qid"]) for row in validation_rows}
    overlaps = {
        "qid": len(train_qids & validation_qids),
        "normalized_question": len(train_questions & validation_questions),
        "answer_entity": len(train_targets & validation_targets),
        "unordered_candidate_set": len(train_candidate_sets & validation_candidate_sets),
    }
    if any(overlaps.values()):
        raise ValueError(f"cross-split identity overlap: {overlaps}")

    train_options = {
        _normalized(option) for row in train_rows for option in row["answer_candidates"]
    }
    validation_options = {
        _normalized(option) for row in validation_rows for option in row["answer_candidates"]
    }
    train_distractors = {
        _normalized(option)
        for row in train_rows
        for option in row["answer_candidates"]
        if _normalized(option) != _normalized(row["answer_entity"])
    }
    target_positions = Counter(
        next(
            index
            for index, option in enumerate(row["answer_candidates"])
            if _normalized(option) == _normalized(row["answer_entity"])
        )
        for row in rows
    )
    train_positions = Counter(
        next(
            index
            for index, option in enumerate(row["answer_candidates"])
            if _normalized(option) == _normalized(row["answer_entity"])
        )
        for row in train_rows
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    def write_jsonl(name: str, split_rows: list[dict[str, Any]]) -> str:
        payload = "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in split_rows
        )
        (output_dir / name).write_text(payload, encoding="utf-8")
        return _sha256(payload.encode("utf-8"))

    train_sha = write_jsonl("train.jsonl", train_rows)
    validation_sha = write_jsonl("validation.jsonl", validation_rows)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": "jaqket-choice-probe-20261124",
        "source": {
            "path": str(input_path),
            "bytes": len(source_bytes),
            "sha256": _sha256(source_bytes),
            "rows": len(rows),
            "format": "JSONL",
            "declared_license": "CC-BY-SA-4.0",
        },
        "split": {
            "algorithm": "answer_entity_question_candidate_set_group_hash_greedy_v1",
            "seed": seed,
            "validation_fraction_target": 0.2,
            "train_rows": len(train_rows),
            "validation_rows": len(validation_rows),
            "group_components": group_components,
            "train_sha256": train_sha,
            "validation_sha256": validation_sha,
            "train_qid_order_sha256": _sha256(
                "\n".join(str(row["qid"]) for row in train_rows).encode()
            ),
            "validation_qid_order_sha256": _sha256(
                "\n".join(str(row["qid"]) for row in validation_rows).encode()
            ),
            "cross_split_overlap": overlaps,
        },
        "integrity": {
            "unique_qid": len({str(row["qid"]) for row in rows}),
            "unique_normalized_question": len({_normalized(row["question"]) for row in rows}),
            "unique_normalized_answer_entity": len(
                {_normalized(row["answer_entity"]) for row in rows}
            ),
            "unique_unordered_candidate_set": len(
                {
                    tuple(sorted(_normalized(option) for option in row["answer_candidates"]))
                    for row in rows
                }
            ),
            "all_options_have_20_choices": True,
            "target_exactly_once_in_options": True,
            "cross_split_option_entity_overlap_count": len(train_options & validation_options),
            "validation_targets_seen_anywhere_in_train_options": len(
                validation_targets & train_options
            ),
            "validation_targets_seen_as_train_distractors": len(
                validation_targets & train_distractors
            ),
            "target_position_counts_by_index": [target_positions[index] for index in range(20)],
            "train_target_position_counts_by_index": [
                train_positions[index] for index in range(20)
            ],
        },
    }
    manifest_path = output_dir / "split-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    manifest = create_split(args.input, args.output_dir, seed=args.seed)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
