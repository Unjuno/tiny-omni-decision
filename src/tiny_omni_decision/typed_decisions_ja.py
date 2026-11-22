"""Parser for the train-only Japanese Typed Decisions development source."""

from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TypedDecisionChoiceGroup:
    sample_id: str
    workflow: str
    question_key: str
    question_type: str
    state_json: str
    instructions: str
    option_labels: tuple[str, ...]
    option_descriptions: tuple[str, ...]
    target_probabilities: tuple[float, ...]
    hard_target: str
    back_translation_match: bool
    split: str


def _normalize(value: Any) -> Any:
    if isinstance(value, str):
        return " ".join(unicodedata.normalize("NFKC", value).split())
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    if isinstance(value, dict):
        return {_normalize(str(key)): _normalize(item) for key, item in value.items()}
    return value


def normalized_state_sha256(state: Any) -> str:
    payload = json.dumps(
        _normalize(state), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _decode_json(value: Any, field: str) -> Any:
    try:
        return json.loads(value) if isinstance(value, str) else value
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field} must contain valid JSON") from exc


def _options(
    question: dict[str, Any], question_type: str
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    criteria = question.get("criteria")
    if isinstance(criteria, dict):
        labels = tuple(str(label) for label in criteria)
        descriptions = tuple(str(criteria[label]) for label in criteria)
    elif isinstance(criteria, list) and question_type == "score":
        labels = tuple(str(index) for index in range(len(criteria)))
        descriptions = tuple(str(description) for description in criteria)
    elif question_type == "noul":
        labels = ("false", "true")
        descriptions = ("文の内容は偽である", "文の内容は真である")
    else:
        raise ValueError("question criteria must define at least two typed options")
    if len(labels) < 2 or len(labels) != len(set(labels)):
        raise ValueError("decision options must have at least two unique labels")
    return labels, descriptions


def parse_typed_decisions_ja_row(row: dict[str, Any]) -> tuple[TypedDecisionChoiceGroup, ...]:
    """Parse one source TRAIN state into its five soft-target choice groups.

    This parser deliberately rejects any row whose publisher split is not
    ``train``. The public test partition is exposed and is never an input to
    this development path.
    """
    if str(row.get("split")) != "train":
        raise ValueError("only publisher train rows are accepted")
    sample_id = str(row.get("id", ""))
    workflow = str(row.get("workflow", ""))
    if not sample_id or not workflow:
        raise ValueError("source id and workflow are required")
    state = _decode_json(row.get("state"), "state")
    questions = _decode_json(row.get("questions"), "questions")
    gold = _decode_json(row.get("gold"), "gold")
    if not isinstance(state, dict) or not isinstance(questions, dict) or not isinstance(gold, dict):
        raise ValueError("state, questions, and gold must be JSON objects")
    if len(questions) != 5 or int(row.get("n_questions", -1)) != 5:
        raise ValueError("each state must contain exactly five questions")
    if set(questions) != set(gold):
        raise ValueError("question and target keys differ")

    state_json = json.dumps(
        _normalize(state), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    back_translation_match = row.get("back_translation_match")
    if not isinstance(back_translation_match, bool):
        raise ValueError("back_translation_match must be boolean")

    groups: list[TypedDecisionChoiceGroup] = []
    for key in sorted(questions):
        question = questions[key]
        target = gold[key]
        if not isinstance(question, dict) or not isinstance(target, dict):
            raise ValueError("question and target entries must be objects")
        question_type = str(question.get("type", ""))
        if question_type not in {"choice", "noul", "score"} or target.get("type") != question_type:
            raise ValueError("unsupported or mismatched question type")
        labels, descriptions = _options(question, question_type)
        probabilities = target.get("probabilities")
        if not isinstance(probabilities, dict) or set(map(str, probabilities)) != set(labels):
            raise ValueError("soft target labels must exactly match question options")
        values = tuple(float(probabilities[label]) for label in labels)
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("target probabilities must be finite and nonnegative")
        if abs(sum(values) - 1.0) > 0.02:
            raise ValueError("target probability mass must sum to one within 0.02")
        hard_target = str(target.get("label", ""))
        if hard_target not in labels:
            raise ValueError("hard target must be one of the typed options")
        groups.append(
            TypedDecisionChoiceGroup(
                sample_id=sample_id,
                workflow=workflow,
                question_key=str(key),
                question_type=question_type,
                state_json=state_json,
                instructions=str(question.get("instructions", "")),
                option_labels=labels,
                option_descriptions=descriptions,
                target_probabilities=values,
                hard_target=hard_target,
                back_translation_match=back_translation_match,
                split="train",
            )
        )
    return tuple(groups)
