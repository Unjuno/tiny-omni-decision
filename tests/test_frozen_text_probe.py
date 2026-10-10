from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

import torch

from scripts.fetch_clevr4_sample import deterministic_split_sample
from scripts.train_frozen_text_probe import CandidateScorer, _assert_disjoint, _metrics


def _example(record_id: str, question: str = "Is it true?") -> SimpleNamespace:
    return SimpleNamespace(
        source_record_id=record_id,
        state='{"value": 1}',
        question=question,
        options=["true", "false"],
        target="true",
    )


def test_frozen_text_probe_rejects_shared_state_ids():
    with pytest.raises(ValueError, match="overlap"):
        _assert_disjoint([_example("state-1:q1")], [_example("state-1:q2")])


def test_frozen_text_probe_rejects_duplicate_normalized_content():
    with pytest.raises(ValueError, match="overlap"):
        _assert_disjoint([_example("train-state:q1")], [_example("val-state:q1")])


def test_candidate_scorer_returns_one_logit_per_option():
    scorer = CandidateScorer(feature_size=8)
    assert scorer(torch.zeros(3, 8)).shape == (3,)


def test_text_probe_metrics_include_calibration_and_counts():
    metrics = _metrics([torch.tensor([4.0, 0.0]), torch.tensor([0.0, 4.0])], [0, 0])
    assert metrics["count"] == 2
    assert metrics["accuracy"] == 0.5
    assert metrics["nll"] > 0
    assert metrics["brier"] > 0
    assert len(metrics["ece_bin_counts"]) == 15
    assert sum(metrics["ece_bin_counts"]) == 2


def test_clevr4_sample_selection_is_deterministic_and_split_specific():
    annotations = {
        f"image-{index:03}": {"split": "train" if index < 8 else "val"} for index in range(12)
    }
    train_a = deterministic_split_sample(annotations, split="train", limit=4, seed=17)
    train_b = deterministic_split_sample(annotations, split="train", limit=4, seed=17)
    validation = deterministic_split_sample(annotations, split="val", limit=2, seed=17)
    assert train_a == train_b
    assert set(train_a).issubset({f"image-{index:03}" for index in range(8)})
    assert set(validation).issubset({f"image-{index:03}" for index in range(8, 12)})
    assert not set(train_a) & set(validation)
