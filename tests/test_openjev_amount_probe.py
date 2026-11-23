from __future__ import annotations

import json

import pytest

from tiny_omni_decision.openjev_amount_probe import (
    assert_split_integrity,
    format_pair,
    make_family_split,
    sanitize_state,
    validate_rows,
)


def _row(sample_id: str, family: str, *, selected: str = "secret-answer") -> dict[str, object]:
    return {
        "id": sample_id,
        "kind": "choice",
        "question": "Choose the amount.",
        "options": ["wrong", "right"],
        "target": [0.0, 1.0],
        "state_json": json.dumps(
            {
                "text": f"Invoice text {family}",
                "policy": {"currency": "USD"},
                "requested_role": "total",
                "candidates": [{"id": "candidate-secret", "text": selected}],
                "selected": selected,
                "selected_id": "candidate-secret",
            }
        ),
        "metadata_json": json.dumps(
            {"document_family_id": family, "question_id": "amount", "case_id": "private-case"}
        ),
    }


def test_sanitize_state_excludes_candidates_and_answer_fields() -> None:
    clean = sanitize_state(_row("x", "f0")["state_json"])
    assert set(clean) == {"text", "policy", "requested_role"}
    assert "selected" not in json.dumps(clean)
    assert "candidate-secret" not in json.dumps(clean)


def test_format_pair_uses_only_sanitized_state_and_visible_decision_text() -> None:
    row = validate_rows([_row("x", "f0")])[0]
    formatted = "\n".join(
        format_pair(row, option)[0] + format_pair(row, option)[1] for option in row["options"]
    )
    assert "secret-answer" not in formatted
    assert "candidate-secret" not in formatted
    assert "private-case" not in formatted
    assert "Choose the amount." in formatted


def test_family_split_is_deterministic_and_keeps_each_family_together() -> None:
    rows = [_row(f"{family}-{i}", family) for family in range(20) for i in range(2)]
    first = make_family_split(rows, seed=17)
    second = make_family_split(rows, seed=17)
    assert first == second
    normalized = validate_rows(rows)
    assert_split_integrity(normalized, first)
    assert set(first.values()) == {"train", "validation"}
    assert len(first) == 20


def test_split_integrity_rejects_same_family_across_splits() -> None:
    first = _row("x0", "family-a")
    second = _row("x1", "family-b")
    second["state_json"] = first["state_json"]
    rows = validate_rows([first, second])
    with pytest.raises(ValueError, match="overlap crosses"):
        assert_split_integrity(rows, {"family-a": "train", "family-b": "validation"})


def test_validation_rejects_misaligned_target() -> None:
    row = _row("x", "f0")
    row["target"] = [1.0]
    with pytest.raises(ValueError, match="target length"):
        validate_rows([row])
