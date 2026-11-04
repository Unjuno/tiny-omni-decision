from __future__ import annotations

import pytest

from tiny_omni_decision.qasc_same_state import (
    build_same_state_criterion_split,
    make_state_derangement,
)


def _row(state: str, question: str, *, row_id: str, target: str = "option-0"):
    return {
        "id": row_id,
        "source_record_id": f"{state}:record-{row_id}",
        "split": "train",
        "license": "CC-BY-4.0",
        "state": f"Evidence: {state}",
        "question": question,
        "options": [target, *[f"{row_id}-option-{i}" for i in range(1, 8)]],
        "target": target,
    }


def test_same_state_split_holds_a_new_question_and_keeps_state_in_training():
    rows = [
        _row("state-a", "Question A1", row_id="a1"),
        _row("state-a", "Question A2", row_id="a2"),
        _row("state-b", "Question B1", row_id="b1"),
        _row("state-b", "Question B2", row_id="b2"),
        _row("state-c", "Question C1", row_id="c1"),
    ]

    train, held, report = build_same_state_criterion_split(rows, seed=17)

    assert len(held) == 2
    assert len(report["held_state_hashes"]) == 2
    assert report["all_held_states_are_in_train"]
    assert report["normalized_question_overlap"] == 0
    assert report["normalized_question_option_overlap"] == 0
    assert {rows[i]["state"] for i in train} >= {rows[i]["state"] for i in held}


def test_same_state_split_is_deterministic():
    rows = [
        _row("state-a", "Question A1", row_id="a1"),
        _row("state-a", "Question A2", row_id="a2"),
        _row("state-b", "Question B1", row_id="b1"),
        _row("state-b", "Question B2", row_id="b2"),
    ]

    first = build_same_state_criterion_split(rows, seed=17)
    second = build_same_state_criterion_split(rows, seed=17)

    assert first == second


def test_same_question_shared_by_states_is_not_selected_for_holding():
    rows = [
        _row("state-a", "Question A1", row_id="a1"),
        _row("state-a", "Question shared", row_id="a2"),
        _row("state-b", "Question B1", row_id="b1"),
        _row("state-b", "Question shared", row_id="b2"),
    ]

    _, held, _ = build_same_state_criterion_split(rows, seed=17)

    assert all(rows[index]["question"] != "Question shared" for index in held)


def test_split_rejects_nontrain_rows_and_reused_ids():
    with pytest.raises(ValueError, match="TRAIN rows only"):
        build_same_state_criterion_split(
            [_row("state-a", "Q1", row_id="1") | {"split": "validation"}]
        )
    duplicate = _row("state-a", "Q1", row_id="same")
    rows = [duplicate, dict(duplicate)]
    with pytest.raises(ValueError, match="unique"):
        build_same_state_criterion_split(rows)


def test_state_derangement_is_repeatable_and_has_no_self_mapping():
    keys = ["state-a", "state-b", "state-c", "state-d"]

    first = make_state_derangement(keys, seed=18)
    second = make_state_derangement(keys, seed=18)

    assert first == second
    assert set(first) == set(first.values()) == set(keys)
    assert all(source != destination for source, destination in first.items())


def test_state_derangement_rejects_duplicate_or_single_state_keys():
    with pytest.raises(ValueError):
        make_state_derangement(["only"])
    with pytest.raises(ValueError):
        make_state_derangement(["same", "same"])


def test_raw_distinct_options_are_not_rejected_for_normalized_similarity():
    first = _row("state-a", "Question A1", row_id="a1")
    first["options"][1] = "A1-OPTION-1"
    second = _row("state-a", "Question A2", row_id="a2")

    train, held, report = build_same_state_criterion_split([first, second], seed=17)

    assert len(train) == len(held) == 1
    assert report["all_held_states_are_in_train"]
