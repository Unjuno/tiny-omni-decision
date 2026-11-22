from __future__ import annotations

import torch

from scripts.train_massive_ja_ruri70m_readout import encode_utterances


class _Tokenizer:
    def __call__(self, texts, **kwargs):
        assert texts == ["first", "second"]
        return {
            "input_ids": torch.ones((2, 3), dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.long),
        }


class _Encoder(torch.nn.Module):
    def forward(self, input_ids, attention_mask):
        values = input_ids.float().unsqueeze(-1).expand(-1, -1, 4)
        return type("Output", (), {"last_hidden_state": values})()


def test_ruri_encoding_uses_masked_mean_then_l2_normalization() -> None:
    encoder = _Encoder()
    rows = [{"utterance": "first"}, {"utterance": "second"}]

    features = encode_utterances(rows, _Tokenizer(), encoder, batch_size=2)

    assert tuple(features.shape) == (2, 4)
    assert features.device.type == "cpu"
    assert torch.isfinite(features).all()
    assert torch.allclose(features.norm(dim=-1), torch.ones(2))
    assert not encoder.training
