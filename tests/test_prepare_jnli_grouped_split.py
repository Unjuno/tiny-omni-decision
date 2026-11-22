from __future__ import annotations

import pytest

from scripts.prepare_jnli_grouped_split import (
    build_grouped_split,
    parse_image_ids,
)


def _row(
    pair_id: int,
    image_pair: str,
    premise: str,
    hypothesis: str,
    *,
    source_split: str = "publisher_train",
) -> dict[str, object]:
    return {
        "sentence_pair_id": str(pair_id),
        "yjcaptions_id": image_pair,
        "sentence1": premise,
        "sentence2": hypothesis,
        "label": ("entailment", "contradiction", "neutral")[pair_id % 3],
        "source_split": source_split,
        "source_row": pair_id + 1,
    }


def test_parse_image_ids_supports_one_or_two_source_images() -> None:
    assert parse_image_ids("123-45-g1") == ("123",)
    assert parse_image_ids("123_456-45-78") == ("123", "456")


@pytest.mark.parametrize("value", ["bad", "1_2_3-a-b", "a-b-c"])
def test_parse_image_ids_rejects_malformed_source_keys(value: str) -> None:
    with pytest.raises(ValueError, match="malformed|non-numeric"):
        parse_image_ids(value)


def test_grouped_split_is_deterministic_and_keeps_image_and_sentence_identity_together() -> None:
    publisher_train = [
        _row(0, "1-a-b", "同じ premise", "hypothesis A"),
        _row(1, "1-c-d", "別の文", "同じ hypothesis"),
        _row(2, "2-e-f", "独立 premise", "独立 hypothesis"),
        _row(3, "3-g-h", "独立 premise 3", "独立 hypothesis 3"),
        _row(4, "4-i-j", "独立 premise 4", "独立 hypothesis 4"),
    ]
    publisher_dev = [
        _row(5, "5-k-l", "同じ　premise", "新しい hypothesis", source_split="publisher_dev"),
        _row(
            6,
            "6_7-m-n",
            "two-image premise",
            "two-image hypothesis",
            source_split="publisher_dev",
        ),
        _row(7, "8-o-p", "独立 premise 8", "独立 hypothesis 8", source_split="publisher_dev"),
    ]
    train, validation, receipt = build_grouped_split(
        publisher_train, publisher_dev, seed=17, validation_fraction=0.375
    )
    train_again, validation_again, receipt_again = build_grouped_split(
        publisher_train, publisher_dev, seed=17, validation_fraction=0.375
    )

    assert train == train_again
    assert validation == validation_again
    assert receipt == receipt_again
    assert receipt["group_overlap"] == 0
    assert receipt["underlying_image_overlap"] == 0
    assert receipt["normalized_sentence_overlap"] == 0
    split_by_group = {row["content_group_id"]: "train" for row in train}
    split_by_group.update({row["content_group_id"]: "validation" for row in validation})
    assert (
        split_by_group[publisher_train[0]["content_group_id"]]
        == split_by_group[publisher_dev[0]["content_group_id"]]
    )
    assert len(train) + len(validation) == 8
