import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

import torch
from torch import nn

from scripts.train_frozen_audio_multiquery import (
    QUESTION_TASKS,
    _batch_logits,
    _pair_features,
    build_query_specs,
)


def test_each_audio_event_gets_four_deterministic_query_types():
    specs = build_query_specs("left")

    assert tuple(spec["question_type"] for spec in specs) == QUESTION_TASKS
    assert specs[0]["options"] == [
        "yes",
        "no",
        "up",
        "down",
        "left",
        "right",
        "on",
        "off",
        "stop",
        "go",
    ]
    assert specs[1]["target_index"] == 0
    assert specs[2]["target_index"] == 1
    assert specs[3]["target_index"] == 1
    assert build_query_specs("yes")[2]["target_index"] == 0
    assert build_query_specs("on")[3]["target_index"] == 0


def test_audio_question_option_features_preserve_option_order():
    audio = torch.tensor([1.0, 0.0])
    question = torch.tensor([0.0, 1.0])
    candidates = torch.tensor([[1.0, 1.0], [-1.0, 1.0]])

    features = _pair_features(audio, question, candidates)
    reversed_features = _pair_features(audio, question, candidates.flip(0))

    torch.testing.assert_close(features.flip(0), reversed_features, rtol=0, atol=0)
    assert features.shape == (2, 14)


def test_variable_choice_batches_mask_padding_before_cross_entropy():
    class SumScorer(nn.Module):
        def forward(self, inputs):
            return inputs.sum(dim=-1, keepdim=True)

    batch = [
        {"options": ["a", "b"], "target_index": 1, "features": torch.tensor([[1.0], [2.0]])},
        {
            "options": ["a", "b", "c"],
            "target_index": 0,
            "features": torch.tensor([[3.0], [4.0], [5.0]]),
        },
    ]

    logits, targets = _batch_logits(SumScorer(), batch, torch.device("cpu"))

    assert logits.shape == (2, 3)
    assert logits[0, 2] == torch.finfo(logits.dtype).min
    assert targets.tolist() == [1, 0]


def test_cpu_smoke_update_keeps_logits_and_gradients_finite():
    scorer = nn.Linear(14, 1)
    optimizer = torch.optim.AdamW(scorer.parameters(), lr=1e-3)
    batch = [
        {
            "options": ["a", "b"],
            "target_index": 1,
            "features": torch.tensor([[0.0] * 14, [1.0] * 14]),
        }
    ]

    logits, targets = _batch_logits(scorer, batch, torch.device("cpu"))
    loss = torch.nn.functional.cross_entropy(logits, targets)
    loss.backward()

    assert torch.isfinite(logits).all()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(parameter.grad).all() for parameter in scorer.parameters())
    optimizer.step()
