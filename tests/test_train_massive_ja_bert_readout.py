from __future__ import annotations

import torch

from scripts.train_massive_ja_bert_readout import classification_metrics, encode_utterances


class _Tokenizer:
    def __call__(self, texts, **kwargs):
        assert texts == ["first", "second"]
        assert kwargs["truncation"] is True
        assert kwargs["max_length"] == 128
        return {"input_ids": torch.ones((len(texts), 2), dtype=torch.long)}


class _Encoder(torch.nn.Module):
    def forward(self, input_ids):
        return type("Output", (), {"last_hidden_state": input_ids.float().unsqueeze(-1)})()


def test_encode_utterances_emits_finite_cpu_cls_features() -> None:
    encoder = _Encoder()
    rows = [{"utterance": "first"}, {"utterance": "second"}]

    features = encode_utterances(rows, _Tokenizer(), encoder, batch_size=2)

    assert tuple(features.shape) == (2, 1)
    assert features.device.type == "cpu"
    assert torch.isfinite(features).all()
    assert not encoder.training


def test_classification_metrics_reports_calibration_and_all_intents() -> None:
    targets = torch.tensor([0, 1])
    logits = torch.full((2, 60), -4.0)
    logits[0, 0] = 4.0
    logits[1, 1] = 4.0

    result = classification_metrics(logits, targets)

    assert result["count"] == 2
    assert result["accuracy"] == 1.0
    assert result["intents_present"] == 2
    assert result["per_intent"]["datetime_query"]["count"] == 1
    assert result["nll"] > 0
    assert result["brier"] > 0
    assert 0 <= result["ece_15_bins"] <= 1
