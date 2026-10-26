"""Small, provenance-preserving QASC-to-Decision row adapter."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from typing import Any


def _normalized(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(re.findall(r"\w+", value, flags=re.UNICODE))


def normalize_qasc_row(
    row: dict[str, Any],
    *,
    revision: str,
    split: str,
    license_name: str = "CC-BY-4.0",
) -> dict[str, Any]:
    """Return a generic text Decision row from one labeled QASC record.

    Supporting facts are used only to derive a stable source-group identity;
    they are intentionally omitted from the model's state, question, and options.
    """
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("revision must be a lowercase 40-character Git commit")
    if split not in {"train", "validation"}:
        raise ValueError("QASC preparation accepts only train or validation rows")
    row_id = row.get("id")
    question = row.get("question")
    if not isinstance(row_id, str) or not row_id.strip():
        raise ValueError("QASC row needs a nonempty id")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("QASC row needs a nonempty question")

    choices = row.get("choices")
    if not isinstance(choices, dict):
        raise ValueError("QASC choices must contain label and text arrays")
    labels, texts = choices.get("label"), choices.get("text")
    if (
        not isinstance(labels, list)
        or not isinstance(texts, list)
        or len(labels) != 8
        or len(texts) != 8
        or any(not isinstance(label, str) or not label.strip() for label in labels)
        or any(not isinstance(text, str) or not text.strip() for text in texts)
        or len(set(labels)) != 8
        or set(labels) != set("ABCDEFGH")
    ):
        raise ValueError("QASC requires eight unique labeled, nonempty choices")
    if len(set(texts)) != 8:
        raise ValueError("QASC choices must be unique as supplied by the source")
    answer_key = row.get("answerKey")
    if not isinstance(answer_key, str) or answer_key not in labels:
        raise ValueError("QASC answerKey must match exactly one supplied choice label")

    facts = (row.get("fact1"), row.get("fact2"))
    if any(not isinstance(fact, str) or not fact.strip() for fact in facts):
        raise ValueError("QASC rows need both source facts for group integrity")
    fact_pair = sorted(_normalized(fact) for fact in facts)
    group_id = hashlib.sha256(
        json.dumps(fact_pair, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    return {
        "id": row_id,
        "source_record_id": f"{group_id}:{row_id}",
        "state": "",
        "question": question,
        "options": texts.copy(),
        "target": texts[labels.index(answer_key)],
        "split": split,
        "license": license_name,
        "commercial_use": True,
        "derivative_model_training_allowed": True,
        "redistribution_allowed": True,
        "attribution": "AI2 QASC dataset authors, CC-BY-4.0; cite Khot et al. (AAAI 2020).",
    }


def check_qasc_split_integrity(
    train_rows: list[dict[str, Any]], validation_rows: list[dict[str, Any]]
) -> dict[str, int]:
    """Reject QASC train/validation overlap by ID, question, fact pair, or content."""

    def identities(
        rows: list[dict[str, Any]], expected_split: str
    ) -> tuple[set[str], set[str], set[str], set[str]]:
        groups: set[str] = set()
        row_ids: set[str] = set()
        questions: set[str] = set()
        contents: set[str] = set()
        for row in rows:
            if row.get("split") != expected_split:
                raise ValueError(f"expected all rows in {expected_split} split")
            source_record_id = row.get("source_record_id")
            if not isinstance(source_record_id, str) or ":" not in source_record_id:
                raise ValueError("QASC rows need a fact-pair group and source row ID")
            group_id, row_id = source_record_id.split(":", 1)
            if not group_id or not row_id or row_id in row_ids:
                raise ValueError("QASC split contains a missing or duplicate identity")
            row_ids.add(row_id)
            groups.add(group_id)
            question = _normalized(row.get("question", ""))
            options = row.get("options")
            if not question or not isinstance(options, list) or len(options) != 8:
                raise ValueError("QASC rows need a question and eight options")
            questions.add(question)
            content = json.dumps(
                {"question": question, "options": sorted(_normalized(x) for x in options)},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            contents.add(hashlib.sha256(content.encode("utf-8")).hexdigest())
        return groups, row_ids, questions, contents

    train_groups, train_ids, train_questions, train_content = identities(train_rows, "train")
    val_groups, val_ids, val_questions, val_content = identities(
        validation_rows, "validation"
    )
    group_overlap = train_groups & val_groups
    id_overlap = train_ids & val_ids
    question_overlap = train_questions & val_questions
    content_overlap = train_content & val_content
    if group_overlap or id_overlap or question_overlap or content_overlap:
        raise ValueError(
            "QASC split overlap: "
            f"{len(group_overlap)} fact-pair groups, {len(id_overlap)} IDs, "
            f"{len(question_overlap)} normalized questions, "
            f"{len(content_overlap)} normalized contents"
        )
    return {
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "train_fact_pair_groups": len(train_groups),
        "validation_fact_pair_groups": len(val_groups),
        "train_validation_group_overlap": 0,
        "train_validation_id_overlap": 0,
        "train_validation_question_overlap": 0,
        "train_validation_content_overlap": 0,
    }


def prepare_qasc_split_rows(
    train_source_rows: list[dict[str, Any]],
    validation_source_rows: list[dict[str, Any]],
    *,
    revision: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Normalize only the labeled train/validation rows and validate their split."""
    train_rows = [
        normalize_qasc_row(row, revision=revision, split="train") for row in train_source_rows
    ]
    validation_rows = [
        normalize_qasc_row(row, revision=revision, split="validation")
        for row in validation_source_rows
    ]
    audit = check_qasc_split_integrity(train_rows, validation_rows)
    return train_rows, validation_rows, audit\n