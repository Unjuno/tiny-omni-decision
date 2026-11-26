"""Catch private-label routing, wrong relevance labels and split leakage."""

import copy
import json

import pytest

from scripts import condition_relevance_corpus as corpus
from scripts.decision_supervision import digest, state_key

STATE = dict(color="red", texture="star", count="3", shape="star")
OTHER = dict(color="blue", texture="metal", count="7", shape="cube")


def test_rows_keep_all_candidates_and_distinguish_truth_from_relevance():
    rows = corpus.make_rows(STATE, "train", 0)
    assert len(rows) == 16
    assert len({r["id"] for r in rows}) == 16
    for row in rows:
        assert len(row["options"]) == len(set(row["option_ids"])) == 40
        labels = row["supervision"]
        assert sum(labels["fact_true"]) == 4
        assert sum(labels["request_relevant"]) == 10
        assert sum(labels["correct"]) == 1
        assert labels["correct"] == [a and b for a, b in zip(
            labels["fact_true"], labels["request_relevant"], strict=True)]
        assert row["target_id"] == f"{row['field']}:{STATE[row['field']]}"
        if row["field"] == "shape":
            texture = row["option_ids"].index("texture:star")
            assert labels["fact_true"][texture]
            assert not labels["request_relevant"][texture]


def test_public_pairs_ignore_all_private_fields_and_stable_ids():
    row = corpus.make_rows(STATE, "train", 0)[0]
    pairs = corpus.public_pairs(row)
    poisoned = {k: copy.deepcopy(v) for k, v in row.items()}
    for key in set(poisoned) - {"state", "question", "options"}:
        poisoned[key] = "PRIVATE_LABEL_POISON"
    assert corpus.public_pairs(poisoned) == pairs
    assert len(pairs) == 40
    assert all("PRIVATE_LABEL_POISON" not in a + b for a, b in pairs)
    assert pairs[0] == (
        f"Question: {row['question']}\nCandidate answer: {row['options'][0]}",
        "State: " + row["state"],
    )


def test_transfer_crosses_query_and_option_wording_without_changing_targets():
    rows = corpus.make_rows(STATE, "transfer", 0)
    assert len(rows) == 64
    groups = {}
    for row in rows:
        groups.setdefault(row["condition"], []).append(row)
    assert {key: len(value) for key, value in groups.items()} == {
        "seen_query_seen_options": 16, "seen_query_new_options": 16,
        "new_query_seen_options": 16, "new_query_new_options": 16,
    }
    for field in STATE:
        matched = [r for r in rows if r["field"] == field]
        assert len({tuple(r["option_ids"]) for r in matched}) == 1
        assert len({r["target_id"] for r in matched}) == 1
        assert len({r["question"] for r in matched}) == 4
        assert len({tuple(r["options"]) for r in matched}) == 4


def test_reconstructing_old_reservation_excludes_every_old_split():
    universe = [STATE, OTHER, dict(STATE, count="4"), dict(STATE, count="5")]
    ordered = sorted(universe, key=lambda s: digest([
        17, "decision-supervision-binding-v1", state_key(s)]))
    available, reserved = corpus.unused_states(universe, [], [1, 1, 1])
    assert reserved == {state_key(s) for s in ordered[:3]}
    assert available == ordered[3:]
    with pytest.raises(ValueError, match="insufficient"):
        corpus.unused_states(universe, [], [2, 2, 2])


def test_historical_exclusions_are_applied_before_reserving_all_three_splits():
    universe = [dict(STATE, count=str(i)) for i in range(1, 7)]
    # Independent recipe for the pre-existing immutable allocator.
    ordered = sorted(universe, key=lambda s: digest([
        17, "decision-supervision-binding-v1", state_key(s)]))
    available, reserved = corpus.unused_states(
        universe, [state_key(ordered[0]), state_key(ordered[2])], [1, 1, 1])
    assert reserved == {state_key(ordered[i]) for i in (1, 3, 4)}
    assert available == [ordered[5]]


def test_independent_state_groups_and_corrupt_supervision_fail_closed():
    parts = {"train": corpus.make_rows(STATE, "train", 0),
             "validation": corpus.make_rows(OTHER, "validation", 0)}
    result = corpus.validate_parts(parts, [])
    assert result["unique_states"] == 2
    overlapping = dict(parts, validation=corpus.make_rows(STATE, "validation", 0))
    with pytest.raises(ValueError, match="overlap"):
        corpus.validate_parts(overlapping, [])
    corrupt = copy.deepcopy(parts)
    corrupt["train"][0]["supervision"]["request_relevant"][0] ^= True
    with pytest.raises(ValueError, match="supervision"):
        corpus.validate_parts(corrupt, [])
    with pytest.raises(ValueError, match="reserved"):
        corpus.validate_parts(parts, [state_key(STATE)])


def test_fit_loader_rejects_transfer_and_calibration_before_any_file_access(tmp_path):
    for purpose in ("transfer", "calibration", "../transfer", "sealed_audit"):
        with pytest.raises(ValueError, match="development partition"):
            corpus.load_development(tmp_path / "does-not-exist", purpose, "0" * 64)


def test_calibration_and_validation_question_banks_are_disjoint():
    banks = {}
    for split in ("train", "calibration", "validation", "transfer"):
        rows = [r for i in range(3) for r in corpus.make_rows(STATE, split, i)]
        banks[split] = {r["question"] for r in rows
                        if r["condition"] != "seen_query_seen_options"
                        and r["condition"] != "seen_query_new_options"}
    for a, b in (("train", "calibration"), ("train", "validation"),
                 ("train", "transfer"), ("validation", "transfer"),
                 ("calibration", "transfer")):
        assert not banks[a] & banks[b]
    assert len(banks["train"]) == 24
    assert len(banks["validation"]) == 8
    assert json.loads(corpus.make_rows(STATE, "train", 0)[0]["state"]) == STATE
