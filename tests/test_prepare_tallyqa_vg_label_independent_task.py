from __future__ import annotations

import pytest

from scripts.prepare_tallyqa_vg_label_independent_task import (
    COUNT_OPTIONS,
    TASK_ID,
    ordered_options,
    prepare,
)


def _screen() -> dict[str, object]:
    train_ids = list(range(10, 18))
    validation_ids = list(range(18, 26))
    return {
        "source": {"repository": "repo", "commit": "commit"},
        "archive_sha256": "archive",
        "split": {
            "train": {"question_ids": train_ids},
            "validation": {"question_ids": validation_ids},
        },
    }


def _acquisition() -> dict[str, object]:
    return {
        "status": "complete",
        "train_receipts": [
            {
                "split": "train",
                "image_id": qid,
                "source_path": f"VG_100K/{qid}.jpg",
                "sha256": f"train-{qid}",
            }
            for qid in range(10, 18)
        ],
        "validation_receipts": [
            {
                "split": "validation",
                "image_id": qid,
                "source_path": f"VG_100K_2/{qid}.jpg",
                "sha256": f"validation-{qid}",
            }
            for qid in range(18, 26)
        ],
    }


def _source_rows() -> list[dict[str, object]]:
    return [
        {
            "data_source": "amt",
            "question_id": qid,
            "image_id": qid,
            "image": f"VG_100K/{qid}.jpg" if qid < 18 else f"VG_100K_2/{qid}.jpg",
            "question": f"How many objects are visible in scene {qid}?",
            "answer": (qid - 10),
        }
        for qid in range(10, 26)
    ]


def test_candidate_vocabulary_and_order_do_not_depend_on_answer() -> None:
    # All possible labels share the same qid-derived candidate ordering.
    candidate_lists = [ordered_options(777) for _answer in COUNT_OPTIONS]
    assert all(values == candidate_lists[0] for values in candidate_lists)
    assert sorted(candidate_lists[0]) == list(COUNT_OPTIONS)
    assert ordered_options(777) == ordered_options(777)
    assert ordered_options(777) != ordered_options(778)


def test_preparation_binds_targets_after_fixed_candidate_order() -> None:
    result = prepare(_screen(), _acquisition(), _source_rows())
    assert result["task"]["id"] == TASK_ID
    assert result["task"]["option_count"] == 16
    assert [row["question_id"] for row in result["train"]] == list(range(10, 18))
    assert [row["question_id"] for row in result["validation"]] == list(range(18, 26))
    for row in result["train"] + result["validation"]:
        values = [option["value"] for option in row["options"]]
        assert sorted(values) == list(COUNT_OPTIONS)
        assert row["options"][row["target"]]["value"] == row["answer"]


def test_preparation_rejects_incomplete_media_receipts() -> None:
    acquisition = _acquisition()
    acquisition["status"] = "incomplete"
    try:
        prepare(_screen(), acquisition, _source_rows())
    except ValueError as error:
        assert "complete media" in str(error)
    else:
        raise AssertionError("incomplete acquisition must fail closed")


def test_preparation_rejects_cross_split_image_identity() -> None:
    screen = _screen()
    acquisition = _acquisition()
    rows = _source_rows()
    rows[-1]["image_id"] = 10
    rows[-1]["image"] = "VG_100K_2/10.jpg"
    acquisition["validation_receipts"].append(
        {
            "split": "validation",
            "image_id": 10,
            "source_path": "VG_100K_2/10.jpg",
            "sha256": "separate-validation-hash",
        }
    )
    with pytest.raises(ValueError, match="image, normalized-question, or media identity"):
        prepare(screen, acquisition, rows)


def test_preparation_rejects_cross_split_normalized_question() -> None:
    rows = _source_rows()
    rows[-1]["question"] = rows[0]["question"].swapcase()
    with pytest.raises(ValueError, match="image, normalized-question, or media identity"):
        prepare(_screen(), _acquisition(), rows)


def test_preparation_rejects_cross_split_media_hash() -> None:
    acquisition = _acquisition()
    acquisition["validation_receipts"][0]["sha256"] = "train-10"
    with pytest.raises(ValueError, match="image, normalized-question, or media identity"):
        prepare(_screen(), acquisition, _source_rows())
