from __future__ import annotations

import pytest

from scripts.prepare_clinc150_near_duplicate_split import build_near_duplicate_split


def make_row(sample_id: str, text: str, label: int, label_name: str) -> dict[str, object]:
    return {"sample_id": sample_id, "text": text, "label": label, "label_name": label_name}


def test_near_duplicates_and_exact_duplicates_stay_together_deterministically() -> None:
    rows = [
        make_row("a", "alpha delta echo", 0, "change_volume"),
        make_row("b", "echo alpha delta", 0, "change_volume"),
        make_row("c", "alpha kappa lambda", 0, "change_volume"),
        make_row("d", "lambda alpha kappa", 0, "change_volume"),
        make_row("e", "bravo foxtrot golf", 1, "bill_balance"),
        make_row("f", "golf bravo foxtrot", 1, "bill_balance"),
        make_row("g", "bravo mike november", 1, "bill_balance"),
        make_row("h", "november bravo mike", 1, "bill_balance"),
        make_row("i", "charlie hotel india", 2, "time"),
        make_row("j", "india charlie hotel", 2, "time"),
        make_row("k", "charlie oscar papa", 2, "time"),
        make_row("l", "papa charlie oscar", 2, "time"),
    ]

    train_a, validation_a, receipt_a = build_near_duplicate_split(rows)
    train_b, validation_b, receipt_b = build_near_duplicate_split(list(reversed(rows)))

    assert train_a == train_b
    assert validation_a == validation_b
    assert receipt_a == receipt_b
    group_splits: dict[str, set[str]] = {}
    for split_name, split_rows in (("train", train_a), ("validation", validation_a)):
        for row in split_rows:
            group_splits.setdefault(str(row["near_duplicate_group_id"]), set()).add(split_name)
    assert all(len(splits) == 1 for splits in group_splits.values())
    assert receipt_a["cross_split_group_overlap"] == 0
    assert receipt_a["near_duplicate_pairs_joined"] >= 2
    assert set(receipt_a["train_class_counts"]) == {"0", "1", "2"}
    assert set(receipt_a["validation_class_counts"]) == {"0", "1", "2"}


def test_cross_label_near_duplicates_remain_in_one_conflict_group() -> None:
    rows = [
        make_row("a", "alpha delta echo", 0, "intent_zero"),
        make_row("b", "echo alpha delta", 1, "intent_one"),
        make_row("c", "alpha kappa lambda", 0, "intent_zero"),
        make_row("d", "lambda alpha kappa", 0, "intent_zero"),
        make_row("e", "alpha romeo sierra", 0, "intent_zero"),
        make_row("f", "sierra alpha romeo", 0, "intent_zero"),
        make_row("g", "bravo foxtrot golf", 1, "intent_one"),
        make_row("h", "golf bravo foxtrot", 1, "intent_one"),
        make_row("i", "bravo mike november", 1, "intent_one"),
        make_row("j", "november bravo mike", 1, "intent_one"),
        make_row("k", "charlie hotel india", 2, "weather"),
        make_row("l", "india charlie hotel", 2, "weather"),
        make_row("m", "charlie oscar papa", 2, "weather"),
        make_row("n", "papa charlie oscar", 2, "weather"),
    ]

    train, validation, receipt = build_near_duplicate_split(rows)

    merged = train + validation
    name_groups: dict[str, set[int]] = {}
    for row in merged:
        name_groups.setdefault(str(row["near_duplicate_group_id"]), set()).add(int(row["label"]))
    assert {0, 1} in name_groups.values()
    assert receipt["cross_label_group_count"] >= 1
    assert receipt["cross_split_group_overlap"] == 0


def test_rejects_uncovered_class_or_duplicate_sample_ids() -> None:
    rows = [
        make_row("a", "alpha beta gamma", 0, "zero"),
        make_row("b", "delta epsilon zeta", 1, "one"),
    ]
    with pytest.raises(ValueError, match="class coverage"):
        build_near_duplicate_split(rows)

    with pytest.raises(ValueError, match="sample_id"):
        build_near_duplicate_split([rows[0], {**rows[1], "sample_id": "a"}])
