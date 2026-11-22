from __future__ import annotations

import torch

from scripts.train_jnli_tohoku_bert_readout import encode_pairs, metrics


class _Tokenizer:
    def __call__(self, left, right, **kwargs):
        assert left == ["premise"]
        assert right == ["hypothesis"]
        return {"input_ids": torch.ones((1, 2), dtype=torch.long)}


class _Encoder(torch.nn.Module):
    def forward(self, input_ids):
        assert tuple(input_ids.shape) == (1, 2)
        return type("Output", (), {"last_hidden_state": torch.ones((1, 2, 4))})()


def test_encode_pairs_uses_one_joint_pair_and_returns_finite_cls_feature():
    result = encode_pairs(
        [{"sentence1": "premise", "sentence2": "hypothesis"}],
        _Tokenizer(),
        _Encoder(),
        batch_size=1,
        max_length=8,
    )
    assert tuple(result.shape) == (1, 4)
    assert torch.isfinite(result).all()


def test_metrics_reports_three_class_probabilistic_scores():
    logits = torch.tensor([[4.0, 0.0, 0.0], [0.0, 4.0, 0.0]])
    result = metrics(logits, torch.tensor([0, 1]))
    assert result["count"] == 2
    assert result["accuracy"] == 1.0
    assert set(result["per_class"]) == {"entailment", "contradiction", "neutral"}
    assert result["nll"] > 0
    assert result["brier"] > 0
