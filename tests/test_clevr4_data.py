from __future__ import annotations

import io

import pytest

from tiny_omni_decision.clevr4_data import (
    read_training_annotations,
    require_disjoint_media,
    split_records_by_asset,
)


def test_training_annotation_reader_skips_heldout_label_values() -> None:
    payload = (
        b'{"train-image":{"split":"train","color":"red","texture":"rubber",'
        b'"count":2,"shape":"cube"},'
        b'"test-image":{"color":unopened_test_label,"split":"test"}}'
    )

    train_rows, split_counts = read_training_annotations(io.BytesIO(payload))

    assert split_counts == {"train": 1, "test": 1}
    assert train_rows == [
        {
            "id": "train-image",
            "split": "train",
            "color": "red",
            "texture": "rubber",
            "count": 2,
            "shape": "cube",
        }
    ]


def test_asset_split_is_deterministic_and_keeps_each_asset_together() -> None:
    records = [
        {"id": "image-a-question-1", "asset_id": "image-a"},
        {"id": "image-a-question-2", "asset_id": "image-a"},
        {"id": "image-b-question-1", "asset_id": "image-b"},
        {"id": "image-c-question-1", "asset_id": "image-c"},
    ]

    first_train, first_validation = split_records_by_asset(
        records, validation_fraction=0.34, seed=17
    )
    second_train, second_validation = split_records_by_asset(
        records, validation_fraction=0.34, seed=17
    )

    assert (first_train, first_validation) == (second_train, second_validation)
    assert set(first_train).isdisjoint(first_validation)
    assert set(first_train) | set(first_validation) == {row["id"] for row in records}
    assert first_validation == ["image-b-question-1", "image-c-question-1"]
    assert first_train == ["image-a-question-1", "image-a-question-2"]


def test_media_identity_overlap_is_rejected_even_when_asset_ids_differ() -> None:
    identical_content = "a" * 64
    train = [{"id": "train-image", "media_sha256": identical_content}]
    validation = [{"id": "validation-image", "media_sha256": identical_content}]

    with pytest.raises(ValueError, match="media identity overlap"):
        require_disjoint_media(train, validation)
