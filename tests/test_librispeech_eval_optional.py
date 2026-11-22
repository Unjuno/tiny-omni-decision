from __future__ import annotations

import pytest


def test_evaluate_single_condition_has_finite_metrics_and_zero_count_for_missing_group():
    torch = pytest.importorskip("torch")
    pytest.importorskip("av")
    pytest.importorskip("transformers")

    from scripts.train_librispeech_frozen_readout import CONDITIONS, _evaluate
    from tiny_omni_decision.decision import FrozenFeatureCandidateScorer

    row = {
        "sample_id": "sample-1",
        "speaker_id": "speaker-1",
        "chapter_id": "chapter-1",
        "condition": "hard_distractors",
        "target_index": 0,
        "options": ["one", "two", "three", "four"],
    }
    scorer = FrozenFeatureCandidateScorer(7 * 384, hidden_size=128)
    metrics, predictions = _evaluate(
        scorer,
        [row],
        {"sample-1": torch.zeros(384)},
        {
            "Question: Which exact transcript matches the spoken audio?": torch.zeros(384),
            **{f"Candidate answer: {name}.": torch.zeros(384) for name in row["options"]},
        },
        torch.device("cpu"),
        4,
    )
    assert set(metrics["by_condition"]) == {*CONDITIONS, "all"}
    assert metrics["by_condition"]["random_distractors"]["count"] == 0
    assert metrics["by_condition"]["random_distractors"]["nll"] is None
    assert metrics["by_condition"]["hard_distractors"]["count"] == 1
    assert torch.isfinite(torch.tensor(metrics["macro_condition"]["nll"]))
    assert len(predictions) == 1
