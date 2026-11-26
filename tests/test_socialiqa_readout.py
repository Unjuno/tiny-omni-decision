from __future__ import annotations

import hashlib

import pytest

from tiny_omni_decision.socialiqa_readout import (
    context_hash,
    validate_source_content_disjoint,
    validate_splits,
)


def example(sample_id: str, context: str, question: str, options: list[str], target: int) -> dict:
    return {
        "sample_id": sample_id,
        "context": context,
        "question": question,
        "options": options,
        "target": target,
        "source_context_sha256": context_hash(context),
    }


def test_validate_splits_accepts_disjoint_contexts_and_counts_targets() -> None:
    train = [example("a", "A person waves.", "Why?", ["hello", "leave", "sleep"], 0)]
    validation = [example("b", "A child eats.", "What next?", ["play", "run", "rest"], 1)]
    report = validate_splits(train, validation)
    assert report["context_group_overlap"] == 0
    assert report["normalized_full_example_overlap"] == 0
    assert report["target_counts"] == {"train": {0: 1}, "validation": {1: 1}}


def test_validate_splits_rejects_context_group_leakage() -> None:
    train = [example("a", "Same context", "Why?", ["x", "y", "z"], 0)]
    validation = [example("b", "  SAME   context ", "Why not?", ["u", "v", "w"], 2)]
    with pytest.raises(ValueError, match="context identity"):
        validate_splits(train, validation)


def test_validate_splits_rejects_question_options_leakage_without_context_overlap() -> None:
    train = [example("a", "Context one", "Why?", ["x", "y", "z"], 0)]
    validation = [example("b", "Context two", "WHY?", ["X", "Y", "Z"], 1)]
    with pytest.raises(ValueError, match="question/options"):
        validate_splits(train, validation)


def test_validate_splits_rejects_option_permuted_question_overlap() -> None:
    train = [example("a", "Context one", "Why?", ["x", "y", "z"], 0)]
    validation = [example("b", "Context two", "WHY?", ["z", "x", "y"], 1)]
    with pytest.raises(ValueError, match="question/options"):
        validate_splits(train, validation)


def test_validate_splits_rejects_conflicting_targets_for_duplicate_question() -> None:
    train = [
        example("a", "Same context", "Why?", ["x", "y", "z"], 0),
        example("b", "same   context", "WHY?", ["z", "x", "y"], 0),
        example("c", "SAME CONTEXT", "why?", ["y", "x", "z"], 1),
    ]
    with pytest.raises(ValueError, match="conflicting targets.*train: 1 groups"):
        validate_splits(train, [])


def test_validate_splits_reports_consistent_duplicate_questions() -> None:
    train = [
        example("a", "Same context", "Why?", ["x", "y", "z"], 0),
        example("b", "same   context", "WHY?", ["z", "x", "y"], 1),
    ]
    report = validate_splits(train, [])
    assert report["duplicate_question_groups"] == {"train": 1, "validation": 0}
    assert report["duplicate_question_excess_rows"] == {"train": 1, "validation": 0}


def test_validate_splits_rejects_forged_context_hash() -> None:
    row = example("a", "Context", "Why?", ["x", "y", "z"], 0)
    row["source_context_sha256"] = hashlib.sha256(b"different").hexdigest()
    with pytest.raises(ValueError, match="context hash"):
        validate_splits([row], [])


def test_source_content_audit_accepts_disjoint_unlabeled_publisher_rows() -> None:
    train = [
        {
            "context": "A person cooks.",
            "question": "Why?",
            "answerA": "Hungry",
            "answerB": "Cold",
            "answerC": "Sleep",
        }
    ]
    dev = [
        {
            "context": "A child runs.",
            "question": "What next?",
            "answerA": "Stop",
            "answerB": "Fall",
            "answerC": "Rest",
        }
    ]

    assert validate_source_content_disjoint(train, dev) == {
        "normalized_context": 0,
        "normalized_full_example": 0,
        "normalized_question_options": 0,
    }


def test_source_content_audit_rejects_context_overlap_without_labels() -> None:
    train = [
        {
            "context": "Same story",
            "question": "Why?",
            "answerA": "A",
            "answerB": "B",
            "answerC": "C",
        }
    ]
    dev = [
        {
            "context": " SAME   STORY ",
            "question": "Why else?",
            "answerA": "D",
            "answerB": "E",
            "answerC": "F",
        }
    ]

    with pytest.raises(ValueError, match="source content overlap"):
        validate_source_content_disjoint(train, dev)


def test_source_content_audit_rejects_option_permuted_duplicate() -> None:
    train = [
        {
            "context": "A person cooks.",
            "question": "Why?",
            "answerA": "Hungry",
            "answerB": "Cold",
            "answerC": "Sleep",
        }
    ]
    source = [
        {
            "context": "Another person cooks.",
            "question": "WHY?",
            "answerA": "Sleep",
            "answerB": "Hungry",
            "answerC": "Cold",
        }
    ]

    with pytest.raises(ValueError, match="source content overlap"):
        validate_source_content_disjoint(train, source)


def test_metrics_on_uniform_logits_are_finite_and_calibrated() -> None:
    pytest.importorskip("torch")
    import torch

    from tiny_omni_decision.socialiqa_readout import classification_metrics

    metrics = classification_metrics(torch.zeros((3, 3)), torch.tensor([0, 1, 2]))
    assert metrics["accuracy"] == pytest.approx(1 / 3)
    assert metrics["macro_accuracy_by_correct_option_position"] == pytest.approx(1 / 3)
    assert metrics["nll"] == pytest.approx(torch.log(torch.tensor(3.0)).item())
    assert metrics["brier"] == pytest.approx(2 / 3)
    assert metrics["ece_15"] == pytest.approx(0.0)
    assert sum(metrics["ece_bin_counts"]) == 3
