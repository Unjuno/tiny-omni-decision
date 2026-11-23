from __future__ import annotations

import hashlib

import pytest

from tiny_omni_decision.socialiqa_readout import context_hash, validate_splits


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


def test_validate_splits_rejects_forged_context_hash() -> None:
    row = example("a", "Context", "Why?", ["x", "y", "z"], 0)
    row["source_context_sha256"] = hashlib.sha256(b"different").hexdigest()
    with pytest.raises(ValueError, match="context hash"):
        validate_splits([row], [])


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
