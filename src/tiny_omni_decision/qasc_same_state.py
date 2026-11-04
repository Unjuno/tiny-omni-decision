"""Deterministic same-state/new-question split helpers for QASC TRAIN only."""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from typing import Any

from .qasc import _normalized


def _digest_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _ordered_ids_sha256(rows: list[dict[str, Any]], indices: list[int]) -> str:
    payload = "".join(f"{rows[index]['id']}\n" for index in indices)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_same_state_criterion_split(
    rows: list[dict[str, Any]], *, seed: int = 17
) -> tuple[list[int], list[int], dict[str, Any]]:
    """Hold one globally unique question per reusable fact-pair state.

    The state must remain represented in training through other questions. This
    is a within-state criterion-transfer diagnostic, not unseen-state testing.
    """
    if not rows:
        raise ValueError("QASC TRAIN rows are required")
    state_rows: dict[str, list[int]] = defaultdict(list)
    question_states: dict[str, set[str]] = defaultdict(set)
    question_rows: dict[str, list[int]] = defaultdict(list)
    source_groups: dict[str, set[str]] = defaultdict(set)
    source_states: dict[str, set[str]] = defaultdict(set)
    seen_ids: set[str] = set()

    for index, row in enumerate(rows):
        if row.get("split") != "train" or row.get("license") != "CC-BY-4.0":
            raise ValueError("same-state split accepts CC-BY-4.0 QASC TRAIN rows only")
        row_id = row.get("id")
        question = row.get("question")
        state = row.get("state")
        options = row.get("options")
        target = row.get("target")
        source_record_id = row.get("source_record_id")
        if not isinstance(row_id, str) or not row_id or row_id in seen_ids:
            raise ValueError("QASC TRAIN row IDs must be nonempty and unique")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("QASC question must be nonempty")
        if not isinstance(state, str) or not state.strip():
            raise ValueError("QASC state must be nonempty")
        if (
            not isinstance(options, list)
            or len(options) != 8
            or any(not isinstance(option, str) or not option.strip() for option in options)
            or len(set(options)) != 8
            or not isinstance(target, str)
            or target not in options
        ):
            raise ValueError("each row must have eight distinct options including its target")
        if not isinstance(source_record_id, str) or ":" not in source_record_id:
            raise ValueError("QASC source_record_id must carry the fact-pair group")
        seen_ids.add(row_id)
        normalized_state = _normalized(state)
        normalized_question = _normalized(question)
        state_key = _digest_text(normalized_state)
        source_group = source_record_id.split(":", 1)[0]
        state_rows[state_key].append(index)
        question_states[normalized_question].add(state_key)
        question_rows[normalized_question].append(index)
        source_groups[state_key].add(source_group)
        source_states[source_group].add(state_key)

    if any(len(groups) != 1 for groups in source_groups.values()):
        raise ValueError("normalized state must map to exactly one source fact-pair group")
    if any(len(groups) != 1 for groups in source_states.values()):
        raise ValueError("source fact-pair group must map to exactly one normalized state")

    state_questions: dict[str, set[str]] = defaultdict(set)
    for question, group_ids in question_states.items():
        for group_id in group_ids:
            state_questions[group_id].add(question)

    eligible: list[tuple[str, list[str]]] = []
    for state_key in sorted(state_rows):
        questions = sorted(state_questions[state_key])
        if len(questions) < 2:
            continue
        globally_unique = [
            question for question in questions if len(question_states[question]) == 1
        ]
        if globally_unique:
            eligible.append((state_key, globally_unique))

    rng = random.Random(seed)
    held_question_by_state = {
        state_key: candidates[rng.randrange(len(candidates))] for state_key, candidates in eligible
    }
    held_set = {
        index
        for question, indices in question_rows.items()
        if any(
            held_question_by_state.get(state_key) == question
            for state_key in question_states[question]
        )
        for index in indices
    }
    held_indices = sorted(held_set)
    train_indices = [index for index in range(len(rows)) if index not in held_set]
    if not held_indices or not train_indices:
        raise ValueError("same-state criterion split must have train and held rows")

    train_questions = {_normalized(rows[index]["question"]) for index in train_indices}
    held_questions = {_normalized(rows[index]["question"]) for index in held_indices}
    if train_questions & held_questions:
        raise ValueError("normalized question leakage across train and held criteria")
    train_question_options = {
        (
            _normalized(rows[index]["question"]),
            tuple(sorted(_normalized(option) for option in rows[index]["options"])),
        )
        for index in train_indices
    }
    held_question_options = {
        (
            _normalized(rows[index]["question"]),
            tuple(sorted(_normalized(option) for option in rows[index]["options"])),
        )
        for index in held_indices
    }
    if train_question_options & held_question_options:
        raise ValueError("normalized question/option content leakage")

    train_state_keys = {_digest_text(_normalized(rows[i]["state"])) for i in train_indices}
    held_state_keys = {_digest_text(_normalized(rows[i]["state"])) for i in held_indices}
    if not held_state_keys.issubset(train_state_keys):
        raise ValueError("every held criterion state must be represented in training")
    if any(len(question_states[q]) != 1 for q in held_questions):
        raise ValueError("held criterion questions must be globally unique to one state")

    report = {
        "schema_version": 1,
        "seed": seed,
        "source_rows": len(rows),
        "state_groups": len(state_rows),
        "states_with_multiple_questions": sum(
            len(questions) >= 2 for questions in state_questions.values()
        ),
        "eligible_states": len(eligible),
        "training_rows": len(train_indices),
        "held_rows": len(held_indices),
        "held_unique_questions": len(held_questions),
        "held_state_groups": len(held_state_keys),
        "train_state_groups": len(train_state_keys),
        "state_overlap_is_intentional": True,
        "all_held_states_are_in_train": True,
        "normalized_question_overlap": 0,
        "normalized_question_option_overlap": 0,
        "train_ids_sha256": _ordered_ids_sha256(rows, train_indices),
        "held_ids_sha256": _ordered_ids_sha256(rows, held_indices),
        "held_question_order_sha256": _digest_text(
            "".join(f"{rows[i]['id']}\t{_normalized(rows[i]['question'])}\n" for i in held_indices)
        ),
        "held_state_hashes": sorted(held_state_keys),
        "held_row_ids": [rows[index]["id"] for index in held_indices],
        "held_question_ids": sorted(held_questions),
        "held_target_in_evidence_count": sum(
            any(
                _normalized(rows[index]["target"]) in _normalized(fact)
                for fact in rows[index]["state"].splitlines()[1:]
            )
            for index in held_indices
        ),
        "held_target_in_evidence_fraction": sum(
            any(
                _normalized(rows[index]["target"]) in _normalized(fact)
                for fact in rows[index]["state"].splitlines()[1:]
            )
            for index in held_indices
        )
        / len(held_indices),
        "official_validation_read": False,
        "official_test_read": False,
        "sealed_audit_read": False,
    }
    return train_indices, held_indices, report


def make_state_derangement(state_keys: list[str], *, seed: int = 18) -> dict[str, str]:
    """Return a deterministic Sattolo derangement across held state identities."""
    ordered = sorted(set(state_keys))
    if len(ordered) < 2 or len(ordered) != len(state_keys):
        raise ValueError("at least two distinct state keys are required")
    destinations = ordered.copy()
    rng = random.Random(seed)
    for index in range(len(destinations) - 1, 0, -1):
        other = rng.randrange(index)
        destinations[index], destinations[other] = destinations[other], destinations[index]
    mapping = dict(zip(ordered, destinations, strict=True))
    if any(source == destination for source, destination in mapping.items()):
        raise AssertionError("Sattolo shuffle unexpectedly contains a fixed point")
    return mapping
