from __future__ import annotations

import json
import zipfile

import pytest

from scripts.screen_tallyqa_vg_train import (
    load_training_rows,
    normalize_question,
    screen_training_rows,
)


def _row(
    question_id: int,
    image_id: int,
    question: str,
    *,
    source: str = "amt",
    image: str | None = None,
    answer: int = 1,
) -> dict[str, object]:
    return {
        "answer": answer,
        "data_source": source,
        "image": image or f"VG_100K/{image_id}.jpg",
        "image_id": image_id,
        "question": question,
        "question_id": question_id,
    }


def test_archive_reader_does_not_open_test_member(tmp_path) -> None:
    archive_path = tmp_path / "annotations.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("train.json", json.dumps([_row(1, 10, "How many cats?")]))
        archive.writestr("test.json", "not valid json and must remain unopened")

    assert len(load_training_rows(archive_path)) == 1


def test_filter_uses_only_amt_rows_with_visual_genome_paths() -> None:
    rows = [
        _row(1, 10, "How many cats?"),
        _row(2, 11, "How many dogs?", image="VG_100K_2/11.jpg"),
        _row(3, 12, "How many birds?", source="imported_genome"),
        _row(
            4,
            13,
            "How many cars?",
            source="imported_vqa",
            image="train2014/COCO_13.jpg",
        ),
    ]

    report = screen_training_rows(rows)

    assert report["filter"]["selected_rows"] == 2
    assert report["filter"]["excluded_rows"] == 2
    assert report["source"]["data_source_counts"] == {
        "amt": 2,
        "imported_genome": 1,
        "imported_vqa": 1,
    }


def test_split_keeps_images_and_repeated_normalized_questions_together() -> None:
    rows = [
        _row(1, 10, "How many cats are there?"),
        _row(2, 11, "HOW MANY CATS ARE THERE!"),
        _row(3, 12, "How many dogs are there?"),
        _row(4, 13, "How many birds are there?"),
        _row(5, 14, "How many chairs are there?"),
        _row(6, 15, "How many cars are there?"),
        _row(7, 16, "How many trees are there?"),
    ]

    first = screen_training_rows(rows, seed=17)
    second = screen_training_rows(rows, seed=17)
    split = first["split"]

    assert first == second
    assert split["image_overlap_count"] == 0
    assert split["normalized_question_overlap_count"] == 0
    assert len(set(split["train"]["image_ids"]) & set(split["validation"]["image_ids"])) == 0
    assert len(set(split["train"]["question_ids"]) & set(split["validation"]["question_ids"])) == 0
    assert len(split["train"]["image_paths"]) == split["train"]["image_count"]
    assert normalize_question("HOW MANY cats are there!") == normalize_question(
        "How many cats are there?"
    )
    locations = {
        image_id: split_name
        for split_name in ("train", "validation")
        for image_id in split[split_name]["image_ids"]
    }
    assert locations[10] == locations[11]


def test_amt_path_conflict_fails_closed() -> None:
    rows = [_row(1, 10, "How many cats?", image="train2014/COCO_10.jpg")]

    with pytest.raises(ValueError, match="allowed VG path"):
        screen_training_rows(rows)
