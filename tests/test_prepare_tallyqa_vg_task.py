from __future__ import annotations

from scripts.prepare_tallyqa_vg_task import option_order, options_for, prepare


def _screen() -> dict[str, object]:
    return {
        "source": {"repository": "repo", "commit": "commit"},
        "archive_sha256": "archive",
        "split": {
            "train": {"question_ids": [10]},
            "validation": {"question_ids": [20]},
        },
    }


def _acquisition() -> dict[str, object]:
    return {
        "status": "complete",
        "train_receipts": [
            {
                "split": "train",
                "image_id": 1,
                "source_path": "VG_100K/1.jpg",
                "sha256": "train-hash",
            }
        ],
        "validation_receipts": [
            {
                "split": "validation",
                "image_id": 2,
                "source_path": "VG_100K_2/2.jpg",
                "sha256": "validation-hash",
            }
        ],
    }


def test_count_options_keep_answer_and_choose_distinct_nearby_values() -> None:
    assert options_for(0) == [0, 1, 2, 3]
    assert options_for(1) == [1, 0, 2, 3]
    assert options_for(9) == [9, 8, 10, 7]
    assert options_for(15) == [15, 14, 13, 12]


def test_option_order_is_deterministic_and_preserves_all_values() -> None:
    assert option_order(123, options_for(5)) == option_order(123, options_for(5))
    assert sorted(option_order(123, options_for(5))) == [3, 4, 5, 6]


def test_preparation_preserves_split_ids_and_media_receipts() -> None:
    source_rows = [
        {
            "data_source": "amt",
            "question_id": 10,
            "image_id": 1,
            "image": "VG_100K/1.jpg",
            "question": "How many objects?",
            "answer": 1,
        },
        {
            "data_source": "amt",
            "question_id": 20,
            "image_id": 2,
            "image": "VG_100K_2/2.jpg",
            "question": "How many objects?",
            "answer": 15,
        },
    ]
    output = prepare(_screen(), _acquisition(), source_rows)
    assert [row["question_id"] for row in output["train"]] == [10]
    assert [row["question_id"] for row in output["validation"]] == [20]
    assert output["train"][0]["media_sha256"] == "train-hash"
    assert output["validation"][0]["target"] == output["validation"][0]["options"].index(
        next(option for option in output["validation"][0]["options"] if option["value"] == 15)
    )


def test_preparation_requires_complete_acquisition() -> None:
    acquisition = _acquisition()
    acquisition["status"] = "incomplete"
    try:
        prepare(_screen(), acquisition, [])
    except ValueError as error:
        assert "complete media" in str(error)
    else:
        raise AssertionError("incomplete media acquisition must fail closed")
