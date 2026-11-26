"""Audit CommonsenseQA train rows and create a question-concept grouped split.

The CLI accepts one pinned training shard only. It never takes validation or
test paths and emits aggregate metadata, hashes, and counts without examples.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SPLIT_SEED = "commonsenseqa-train-question-concept-split-20261126-v1"
VALIDATION_PERCENT = 20
CHOICE_LABELS = frozenset("ABCDE")


def normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def stable_split(concept: str, seed: str = SPLIT_SEED) -> str:
    normalized_concept = normalize(concept)
    if not normalized_concept:
        raise ValueError("question_concept must not be empty")
    digest = hashlib.sha256(f"{seed}\0{normalized_concept}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") % 100
    return "validation" if bucket < VALIDATION_PERCENT else "train"


def _hash_lines(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def _choice_word_counts(row: dict[str, Any]) -> tuple[int, int]:
    labels = row["choices"]["label"]
    texts = row["choices"]["text"]
    correct = labels.index(row["answerKey"])
    return len(normalize(texts[correct]).split()), sum(
        len(normalize(text).split()) for index, text in enumerate(texts) if index != correct
    )


def answer_alias_distribution(row: dict[str, Any]) -> dict[str, float]:
    """Treat text-identical options as equivalent correct labels."""
    labels = [str(value) for value in row["choices"]["label"]]
    texts = [normalize(str(value)) for value in row["choices"]["text"]]
    answer = str(row["answerKey"])
    if answer not in labels:
        raise ValueError("answerKey must identify one of the choice labels")
    answer_text = texts[labels.index(answer)]
    aliases = [label for label, text in zip(labels, texts, strict=True) if text == answer_text]
    return {label: 1.0 / len(aliases) for label in aliases}


def audit_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("training shard is empty")

    ids: list[str] = []
    normalized_questions: list[str] = []
    normalized_mcqs: list[str] = []
    answer_positions: Counter[str] = Counter()
    concept_rows: dict[str, list[int]] = defaultdict(list)
    correct_word_counts: list[int] = []
    distractor_word_counts: list[int] = []
    duplicate_option_records = 0
    correct_alias_records = 0
    duplicate_distractor_records = 0

    for row in records:
        try:
            record_id = str(row["id"])
            concept = str(row["question_concept"])
            question = str(row["question"])
            labels = [str(value) for value in row["choices"]["label"]]
            texts = [str(value) for value in row["choices"]["text"]]
            answer = str(row["answerKey"])
            if (
                not record_id
                or not normalize(concept)
                or not normalize(question)
                or len(labels) != 5
                or set(labels) != CHOICE_LABELS
                or len(texts) != 5
                or any(not normalize(text) for text in texts)
                or answer not in CHOICE_LABELS
            ):
                raise ValueError("record has invalid fields or choice schema")
        except (KeyError, TypeError) as error:
            raise ValueError("record has invalid fields or choice schema") from error

        ids.append(record_id)
        normalized_concept = normalize(concept)
        normalized_question = normalize(question)
        normalized_questions.append(normalized_question)
        normalized_mcqs.append(
            json.dumps(
                [normalized_question, *sorted(normalize(text) for text in texts)],
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
        concept_rows[normalized_concept].append(len(ids) - 1)
        answer_positions[answer] += 1
        normalized_choices = [normalize(text) for text in texts]
        if len(set(normalized_choices)) != len(normalized_choices):
            duplicate_option_records += 1
            if len(answer_alias_distribution(row)) > 1:
                correct_alias_records += 1
            else:
                duplicate_distractor_records += 1
        correct_words, distractor_words = _choice_word_counts(row)
        correct_word_counts.append(correct_words)
        distractor_word_counts.append(distractor_words / 4)

    if len(set(ids)) != len(ids):
        raise ValueError("training shard contains duplicate IDs")

    split_ids: dict[str, list[str]] = {"train": [], "validation": []}
    split_concepts: dict[str, set[str]] = {"train": set(), "validation": set()}
    split_questions: dict[str, set[str]] = {"train": set(), "validation": set()}
    split_mcqs: dict[str, set[str]] = {"train": set(), "validation": set()}
    for concept, row_indices in concept_rows.items():
        split = stable_split(concept)
        split_concepts[split].add(concept)
        for row_index in row_indices:
            split_ids[split].append(ids[row_index])
            split_questions[split].add(normalized_questions[row_index])
            split_mcqs[split].add(normalized_mcqs[row_index])

    concept_overlap = split_concepts["train"] & split_concepts["validation"]
    question_overlap = split_questions["train"] & split_questions["validation"]
    mcq_overlap = split_mcqs["train"] & split_mcqs["validation"]
    if concept_overlap or question_overlap or mcq_overlap:
        raise ValueError("group split produced cross-split overlap")

    duplicate_question_rows = len(normalized_questions) - len(set(normalized_questions))
    duplicate_mcq_rows = len(normalized_mcqs) - len(set(normalized_mcqs))
    return {
        "source_split": "train_only",
        "split_seed": SPLIT_SEED,
        "validation_assignment": (
            "sha256(seed + NUL + normalized question_concept), first 64 bits modulo 100 < 20"
        ),
        "rows": len(records),
        "unique_ids": len(set(ids)),
        "unique_question_concepts": len(concept_rows),
        "normalized_duplicate_question_rows": duplicate_question_rows,
        "normalized_duplicate_mcq_rows": duplicate_mcq_rows,
        "rows_with_duplicate_option_text": duplicate_option_records,
        "rows_with_correct_answer_aliases": correct_alias_records,
        "rows_with_duplicate_distractors_only": duplicate_distractor_records,
        "largest_question_concept_groups": sorted(
            (len(group) for group in concept_rows.values()), reverse=True
        )[:10],
        "split_rows": {name: len(values) for name, values in split_ids.items()},
        "split_concepts": {name: len(values) for name, values in split_concepts.items()},
        "split_id_order_sha256": {
            name: _hash_lines(values) for name, values in split_ids.items()
        },
        "split_concept_set_sha256": {
            name: _hash_lines(sorted(values)) for name, values in split_concepts.items()
        },
        "cross_split_overlap": {
            "question_concepts": len(concept_overlap),
            "normalized_questions": len(question_overlap),
            "normalized_mcqs": len(mcq_overlap),
        },
        "answer_position_counts": dict(sorted(answer_positions.items())),
        "mean_correct_answer_words": sum(correct_word_counts) / len(correct_word_counts),
        "mean_distractor_words": sum(distractor_word_counts) / len(distractor_word_counts),
    }


def audit_train_parquet(path: Path) -> dict[str, Any]:
    lowered_name = path.name.casefold()
    if "train" not in lowered_name or any(
        token in lowered_name for token in ("test", "valid", "dev")
    ):
        raise ValueError("only a training parquet shard is accepted")
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError("pyarrow is required to read the pinned Parquet shard") from error

    receipt = audit_records(parquet.read_table(path).to_pylist())
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    receipt["input"] = {
        "filename": path.name,
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    receipt = audit_train_parquet(args.train)
    rendered = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
