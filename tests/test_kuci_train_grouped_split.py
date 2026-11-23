from __future__ import annotations

import copy

import pytest

from scripts.build_kuci_train_grouped_split import grouped_split


def _row(index: int) -> dict[str, object]:
    return {
        "id": index,
        "context": f"事象 {index} の後に、",
        "choice_a": f"選択肢a-{index}",
        "choice_b": f"選択肢b-{index}",
        "choice_c": f"選択肢c-{index}",
        "choice_d": f"選択肢d-{index}",
        "label": "abcd"[index % 4],
        "agreement": 3,
        "core_event_pair": f"event-{index}",
    }


def test_grouped_split_is_deterministic_and_keeps_all_label_positions() -> None:
    rows = [_row(index) for index in range(40)]
    first = grouped_split(rows, seed=17, validation_fraction=0.2)
    second = grouped_split(rows, seed=17, validation_fraction=0.2)

    assert [row["id"] for row in first[0]] == [row["id"] for row in second[0]]
    assert [row["id"] for row in first[1]] == [row["id"] for row in second[1]]
    assert first[2]["sample_assignment_sha256"] == second[2]["sample_assignment_sha256"]
    assert set(first[2]["train_label_counts"]) == set("abcd")
    assert set(first[2]["validation_label_counts"]) == set("abcd")


@pytest.mark.parametrize("duplicate_key", ["event_pair", "context", "option_set"])
def test_duplicate_source_or_normalized_content_is_kept_in_one_split(duplicate_key: str) -> None:
    rows = [_row(index) for index in range(80)]
    duplicate = copy.deepcopy(rows[0])
    duplicate["id"] = 10_000
    if duplicate_key == "event_pair":
        duplicate["context"] = "別のcontext"
        duplicate["core_event_pair"] = rows[1]["core_event_pair"]
    elif duplicate_key == "context":
        duplicate["context"] = f"\u3000{rows[1]['context']} "
        duplicate["core_event_pair"] = "different-event"
    else:
        duplicate["context"] = rows[2]["context"]
        duplicate["core_event_pair"] = "different-event"
        duplicate["choice_a"], duplicate["choice_b"] = (
            rows[2]["choice_b"],
            rows[2]["choice_a"],
        )
        duplicate["choice_c"], duplicate["choice_d"] = (
            rows[2]["choice_d"],
            rows[2]["choice_c"],
        )
    rows.append(duplicate)

    train, validation, _ = grouped_split(rows, seed=29, validation_fraction=0.2)
    locations = {
        str(row["id"]): "train" if row in train else "validation"
        for row in (rows[0], rows[1], rows[2], duplicate)
    }
    assert (
        locations[str(duplicate["id"])]
        == locations[
            str(
                rows[
                    0 if duplicate_key == "event_pair" else 1 if duplicate_key == "context" else 2
                ]["id"]
            )
        ]
    )
    assert train and validation


def test_grouped_split_fails_closed_on_missing_identity_or_duplicate_ids() -> None:
    missing_group = _row(0)
    missing_group["core_event_pair"] = ""
    with pytest.raises(ValueError, match="core_event_pair"):
        grouped_split([missing_group, _row(1)])

    duplicate_id = _row(0)
    with pytest.raises(ValueError, match="duplicate KUCI id"):
        grouped_split([_row(0), duplicate_id])


def test_grouped_split_rejects_unreasonable_fraction() -> None:
    with pytest.raises(ValueError, match="validation_fraction"):
        grouped_split([_row(index) for index in range(20)], validation_fraction=0.5)
