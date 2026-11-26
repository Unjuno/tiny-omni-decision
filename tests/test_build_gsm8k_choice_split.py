from __future__ import annotations

import json

import pytest

from scripts.build_gsm8k_choice_split import _jsonl_line, build_examples


def _row(question: str, answer: int) -> dict[str, str]:
    return {
        "question": question,
        "answer": f"Step: {answer - 1} + 1 = {answer}\n#### {answer}",
    }


def _fixture() -> list[dict[str, str]]:
    rows = [
        _row(f"Question number {index}: compute {index} plus one.", index + 1)
        for index in range(12)
    ]
    # Exact normalized duplicates are one split group, even when punctuation differs.
    rows.extend(
        [
            _row("Question number 0: compute 0 plus one!", 1),
            _row("Question number 1: compute 1 plus one!", 2),
        ]
    )
    return rows


def test_split_is_deterministic_and_keeps_duplicate_questions_together() -> None:
    rows = _fixture()
    train_a, validation_a, info_a = build_examples(rows, seed=17)
    train_b, validation_b, info_b = build_examples(rows, seed=17)

    assert [item.id for item in train_a] == [item.id for item in train_b]
    assert [item.id for item in validation_a] == [item.id for item in validation_b]
    assert info_a == info_b
    memberships = {item.source_record_id: item.split for item in train_a + validation_a}
    assert memberships
    # The punctuation-normalized duplicates each retain two row IDs, but one group.
    grouped = [
        item for item in train_a + validation_a if "question number 0" in item.question.casefold()
    ]
    assert len(grouped) == 2
    assert len({item.split for item in grouped}) == 1
    assert info_a["official_test_read"] is False


def test_generated_choices_are_four_unique_values_with_one_correct_target() -> None:
    train, validation, _ = build_examples(_fixture(), seed=17)
    assert train and validation
    for example in train + validation:
        assert len(example.options) == 4
        assert len(set(example.options)) == 4
        assert example.target in example.options
        assert example.modality == "text"
        assert example.source_record_id


def test_train_and_validation_problem_identities_do_not_overlap() -> None:
    train, validation, _ = build_examples(_fixture(), seed=17)
    assert {item.source_record_id for item in train}.isdisjoint(
        {item.source_record_id for item in validation}
    )


def test_duplicate_question_with_conflicting_gold_answer_fails_closed() -> None:
    rows = _fixture()
    rows.append(_row("Question number 0: compute 0 plus one?", 999))
    with pytest.raises(ValueError, match="conflicting numeric answers"):
        build_examples(rows, seed=17)


def test_missing_official_answer_marker_fails_closed() -> None:
    rows = _fixture()
    rows[0]["answer"] = "The answer is 1"
    with pytest.raises(ValueError, match="official final-answer marker"):
        build_examples(rows, seed=17)


def test_jsonl_escapes_unicode_line_separator() -> None:
    rows = _fixture()
    rows[0]["question"] = "First line.\u2028Second line."
    train, validation, _ = build_examples(rows, seed=17)
    example = next(item for item in train + validation if "First line" in item.question)
    serialized = _jsonl_line(example)

    assert "\\u2028" in serialized
    assert json.loads(serialized)["question"] == example.question
