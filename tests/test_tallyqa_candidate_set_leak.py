from __future__ import annotations

import pytest

from scripts.audit_tallyqa_candidate_set_leak import summarize_train_rows


def _row(answer: int, values: list[int]) -> dict[str, object]:
    return {
        "answer": answer,
        "options": [{"value": value} for value in values],
    }


def test_legacy_policy_unique_candidate_sets_reveal_counts_three_through_thirteen() -> None:
    summary = summarize_train_rows(
        [
            _row(0, [0, 1, 2, 3]),
            _row(3, [1, 2, 3, 4]),
            _row(4, [2, 3, 4, 5]),
        ]
    )
    assert summary["policy_uniquely_reveals_possible_labels"] == list(range(3, 14))
    assert summary["train_rows_in_sets_with_one_observed_train_label"] == 3
    assert summary["distinct_candidate_sets"] == 3


def test_shortcut_audit_rejects_rows_without_the_answer_in_options() -> None:
    with pytest.raises(ValueError, match="answer is not present"):
        summarize_train_rows([_row(4, [0, 1, 2, 3])])
