import json
from pathlib import Path

import numpy as np
import pytest

from scripts import semantic_state_diagnostic as diagnostic
from scripts.evaluate_semantic_state_bridge import sha, verify


def row(identity="a", **changes):
    return dict(id=identity, split="train", color="red", texture="star", count=5,
                shape="star", **changes)


def test_selection_is_train_only_unique_order_independent_and_prefix_stable():
    rows = [row(str(i)) for i in range(12)]
    chosen = diagnostic.select_train(rows, 8)
    assert chosen == diagnostic.select_train(list(reversed(rows)), 8)
    assert chosen[:4] == diagnostic.select_train(rows, 4)
    for bad in [rows + [rows[0]], [dict(rows[0], split="validation")],
                [dict(rows[0], color="unknown")]]:
        with pytest.raises(ValueError):
            diagnostic.select_train(bad, 1)
    with pytest.raises(ValueError):
        diagnostic.select_train(rows, 13)


def test_same_forty_options_have_distinct_typed_targets_for_four_questions():
    options = diagnostic.options()
    assert len(options) == len({o["id"] for o in options}) == 40
    assert {o["id"] for o in options if o["id"].endswith(":star")} == {
        "texture:star", "shape:star"}
    targets = diagnostic.targets(diagnostic.oracle_state(row()))
    assert [options[t]["id"] for t in targets] == [
        "color:red", "texture:star", "count:5", "shape:star"]
    assert len(set(targets)) == 4


def test_model_pair_contains_only_explicit_state_question_and_candidate():
    state = diagnostic.oracle_state(dict(row(), target="leak", secret="forbidden"))
    assert set(state) == {"color", "texture", "count", "shape"}
    first, second = diagnostic.model_pair(state, "Which shape?", "shape: star")
    assert first == "Question: Which shape?\nCandidate answer: shape: star"
    assert json.loads(second.removeprefix("State: ")) == state
    assert "forbidden" not in second and '"id"' not in second and "leak" not in second
    assert diagnostic.model_pair({}, "", "count: 5") == (
        "Question: \nCandidate answer: count: 5", "State: {}")


def test_predicted_state_depends_only_on_native_logits_not_annotation():
    scores = np.full((4, 10), -10.0)
    scores[np.arange(4), [0, 1, 2, 3]] = 10
    assert diagnostic.predicted_state(scores) == dict(
        color="gray", texture="metal", count="3", shape="cone")
    for bad in [scores[:, :9], np.full((4, 10), np.nan)]:
        with pytest.raises(ValueError):
            diagnostic.predicted_state(bad)


def test_metrics_stable_nll_and_binary_brier_hand_calculation():
    metrics = diagnostic.metrics(np.log([[0.8, 0.2], [0.3, 0.7]]), np.array([0, 0]))
    assert metrics["accuracy"] == 0.5
    assert metrics["nll"] == pytest.approx(-np.log(0.24) / 2)
    assert metrics["brier"] == pytest.approx((0.08 + 0.98) / 2)
    assert metrics["ece"] == pytest.approx(0.45)
    assert diagnostic.metrics(np.array([[0, -1000.0]]), np.array([1]))["nll"] == 1000
    with pytest.raises(ValueError):
        diagnostic.metrics(np.zeros((1, 2)), np.array([2]))


def test_summary_separates_field_routing_value_and_changed_question_pairs():
    target = np.array([[1, 16, 24, 35], [1, 16, 24, 35]])
    scores = np.full((2, 4, 40), -10.0)
    for i in range(4):
        scores[:, i, target[0, i]] = 5
    result = diagnostic.summarize(scores, target)
    assert result["all_four_correct"] == 1
    assert result["changed_question_pair_both_correct"] == 1
    assert result["field_routing_accuracy"] == 1
    assert result["within_correct_field_accuracy"] == 1
    # Question-agnostic prediction chooses color for all four questions.
    scores[:] = -10
    scores[:, :, 1] = 10
    result = diagnostic.summarize(scores, target)
    assert result["overall"]["accuracy"] == 0.25
    assert result["field_routing_accuracy"] == 0.25
    assert result["all_four_correct"] == 0
    assert result["changed_question_pair_both_correct"] == 0
    assert result["pair_count"] == 12


def test_donor_control_rolls_whole_state_and_corrected_targets_together():
    scores = np.arange(3 * 4 * 40).reshape(3, 4, 40).astype(float)
    targets = np.array([[0, 10, 20, 30], [1, 11, 21, 31], [2, 12, 22, 32]])
    swapped, corrected = diagnostic.donor_control(scores, targets)
    assert np.array_equal(swapped[0], scores[1])
    assert np.array_equal(corrected[0], targets[1])
    original = diagnostic.summarize(scores, targets)
    restored = diagnostic.summarize(swapped, corrected)
    assert restored["overall"] == pytest.approx(original["overall"], abs=1e-12)
    for field in diagnostic.FIELDS:
        assert restored["per_question"][field] == pytest.approx(
            original["per_question"][field], abs=1e-12)


def test_pins_fail_closed_on_empty_missing_or_modified_sources(tmp_path):
    source = tmp_path / "source.json"
    source.write_text("original")
    pins = {str(source): sha(source)}
    verify(pins)
    source.write_text("changed")
    with pytest.raises(ValueError, match="source changed"):
        verify(pins)
    with pytest.raises(ValueError, match="empty"):
        verify({})
    with pytest.raises(FileNotFoundError):
        verify({str(tmp_path / "missing"): "0" * 64})


def test_fixed_contract_has_no_validation_inputs_or_runtime_hyperparameter_search():
    config = json.loads(Path(
        "configs/pretrained_reuse/semantic_state_bridge_diagnostic.json").read_text())
    assert config["images"] == 256 and config["smoke_images"] == 4
    assert config["seed"] == 17 and config["pair_batch_size"] == 64
    for path, digest in config["inputs"].items():
        assert len(digest) == 64 and int(digest, 16) >= 0
        assert not any(word in path for word in ("validation", "val-", "audit", "sealed"))
    assert len(config["base_audit_sha256"]) == 64
