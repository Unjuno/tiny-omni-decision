from __future__ import annotations

import pytest

from scripts.compare_jcommonsenseqa_ruri_readouts import _paired_group_bootstrap
from scripts.prepare_jcommonsenseqa_grouped_split import (
    build_grouped_split,
    target_text_is_ambiguous,
)
from scripts.train_jcommonsenseqa_ruri_readout import MODEL_PINS


def test_ruri_readout_accepts_only_fixed_verified_model_revisions() -> None:
    assert MODEL_PINS == {
        "cl-nagoya/ruri-v3-30m": {
            "revision": "24899e5de370b56d179604a007c0d727bf144504",
            "license": "Apache-2.0",
            "weights_sha256": "e94155e342cbfc33280c345c6e2912905fb5de6e1a3adeb938f395aafd2f8832",
        },
        "cl-nagoya/ruri-v3-70m": {
            "revision": "07a8b0aba47d29d2ca21f89b915c1efe2c23d1cc",
            "license": "Apache-2.0",
            "weights_sha256": "c959b76a535eee838b1f1e6b8999ba513ef8d87a6f8c3534c50dcda1eb5f6a62",
        },
    }


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


def test_paired_bootstrap_keeps_question_groups_together_and_is_deterministic() -> None:
    rows = [
        {
            "sample_id": "q-1",
            "content_group_id": "g1",
            "target_index": 0,
            "probabilities": [0.7, 0.075, 0.075, 0.075, 0.075],
        },
        {
            "sample_id": "q-2",
            "content_group_id": "g1",
            "target_index": 1,
            "probabilities": [0.7, 0.075, 0.075, 0.075, 0.075],
        },
        {
            "sample_id": "q-3",
            "content_group_id": "g2",
            "target_index": 0,
            "probabilities": [0.1, 0.7, 0.1, 0.05, 0.05],
        },
    ]
    result = _paired_group_bootstrap(rows, rows, iterations=100, seed=17)
    repeat = _paired_group_bootstrap(rows, rows, iterations=100, seed=17)
    assert result == repeat
    assert result["group_count"] == 2
    assert all(
        value["point_difference"] == 0.0 for value in result["candidate_minus_baseline"].values()
    )


def test_paired_bootstrap_rejects_misaligned_prediction_rows() -> None:
    baseline = [
        {
            "sample_id": "q-1",
            "content_group_id": "g1",
            "target_index": 0,
            "probabilities": [0.2] * 5,
        }
    ]
    candidate = [
        {
            "sample_id": "wrong-id",
            "content_group_id": "g1",
            "target_index": 0,
            "probabilities": [0.2] * 5,
        }
    ]
    with pytest.raises(ValueError, match="do not align"):
        _paired_group_bootstrap(baseline, candidate, iterations=10, seed=17)
