from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[1] / "scripts" / "audit_commonsenseqa_train.py"
SPEC = importlib.util.spec_from_file_location("commonsenseqa_train_audit", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _row(record_id: str, concept: str, question: str, answer: str = "A") -> dict[str, object]:
    return {
        "id": record_id,
        "question_concept": concept,
        "question": question,
        "choices": {
            "label": list("ABCDE"),
            "text": ["correct", "wrong b", "wrong c", "wrong d", "wrong e"],
        },
        "answerKey": answer,
    }


def test_split_is_deterministic_and_keeps_concepts_questions_and_options_together() -> None:
    rows = [
        _row("1", "shared", "first question"),
        _row("2", "shared", "second question", "B"),
        _row("3", "shared", "first question"),
        *[_row(f"{index}", f"concept-{index}", f"question-{index}") for index in range(4, 200)],
    ]

    first = MODULE.audit_records(rows)
    second = MODULE.audit_records(rows)

    assert first == second
    assert first["cross_split_overlap"] == {
        "question_concepts": 0,
        "normalized_questions": 0,
        "normalized_mcqs": 0,
    }
    assert sum(first["split_rows"].values()) == len(rows)
    assert sum(first["answer_position_counts"].values()) == len(rows)


def test_rejects_duplicate_ids_and_malformed_choice_sets() -> None:
    duplicate_ids = [_row("same", "one", "first"), _row("same", "two", "second")]
    with pytest.raises(ValueError, match="duplicate IDs"):
        MODULE.audit_records(duplicate_ids)

    malformed = _row("bad", "one", "first")
    malformed["choices"]["label"] = ["A", "B"]
    with pytest.raises(ValueError, match="invalid fields"):
        MODULE.audit_records([malformed])


def test_exact_text_option_aliases_get_equal_correct_probability_mass() -> None:
    row = _row("alias", "one", "first", answer="B")
    row["choices"]["text"][0] = "wrong b"

    assert MODULE.answer_alias_distribution(row) == {"A": 0.5, "B": 0.5}
    audit = MODULE.audit_records([row])
    assert audit["rows_with_duplicate_option_text"] == 1
    assert audit["rows_with_correct_answer_aliases"] == 1
    assert audit["rows_with_duplicate_distractors_only"] == 0


def test_cli_input_guard_rejects_nontraining_shard(tmp_path: Path) -> None:
    validation = tmp_path / "validation-00000.parquet"
    validation.touch()
    with pytest.raises(ValueError, match="only a training parquet shard"):
        MODULE.audit_train_parquet(validation)
