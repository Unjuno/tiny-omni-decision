from __future__ import annotations

import pytest

from scripts.prepare_jcommonsenseqa_grouped_split import (
    build_grouped_split,
    target_text_is_ambiguous,
)


def row(q_id: int, question: str, label: int = 0) -> dict[str, object]:
    return {
        "q_id": q_id,
        "question": question,
        "choice0": f"answer-{q_id}-0",
        "choice1": f"answer-{q_id}-1",
        "choice2": f"answer-{q_id}-2",
        "choice3": f"answer-{q_id}-3",
        "choice4": f"answer-{q_id}-4",
        "label": label,
        "source_split": "source",
    }


def fixture_rows() -> list[dict[str, object]]:
    rows = [row(index, f"独立の質問 {index}", index % 5) for index in range(20)]
    rows.extend([row(20, "同じ 問題", 0), row(21, "同じ\u3000問題", 1)])
    return rows


def test_grouped_split_is_deterministic_and_keeps_normalized_question_together() -> None:
    rows = fixture_rows()
    train, validation, receipt = build_grouped_split(rows, seed=17)
    train_again, validation_again, receipt_again = build_grouped_split(rows, seed=17)
    assert train == train_again
    assert validation == validation_again
    assert receipt == receipt_again
    split_by_id = {item["q_id"]: "train" for item in train}
    split_by_id.update({item["q_id"]: "validation" for item in validation})
    assert split_by_id[20] == split_by_id[21]
    assert receipt["cross_split_question_group_overlap"] == 0
    assert len(train) + len(validation) == len(rows)


def test_repeated_question_rows_with_conflicting_targets_are_retained() -> None:
    rows = fixture_rows()
    train, validation, receipt = build_grouped_split(rows, seed=17)
    assert len(train) + len(validation) == len(rows)
    assert receipt["conflicting_correct_answer_groups_retained"] == 1


def test_duplicate_choice_text_is_retained_and_duplicate_ids_fail_closed() -> None:
    bad = row(0, "bad")
    bad["choice4"] = bad["choice3"]
    rows = [bad, *[row(i, f"question {i}", i % 5) for i in range(1, 10)]]
    train, validation, receipt = build_grouped_split(rows)
    assert len(train) + len(validation) == len(rows)
    assert receipt["rows_with_duplicate_choice_text_retained"] == 1
    duplicated = fixture_rows()
    duplicated[1]["q_id"] = duplicated[0]["q_id"]
    with pytest.raises(ValueError, match="duplicate q_id"):
        build_grouped_split(duplicated)


def test_target_text_ambiguity_uses_normalized_choice_identity() -> None:
    example = row(0, "ambiguous", label=0)
    example["choice1"] = "Ａnswer-0-0"
    assert target_text_is_ambiguous(example)
