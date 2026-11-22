from __future__ import annotations

import torch

from scripts.train_jcommonsenseqa_tohoku_bert_readout import choice_metrics, encode_options


class _Tokenizer:
    def __call__(self, questions, choices, **kwargs):
        assert questions == ["question"] * len(choices)
        assert all(choice.startswith("candidate ") for choice in choices)
        return {"input_ids": torch.ones((len(choices), 2), dtype=torch.long)}


class _Encoder(torch.nn.Module):
    def forward(self, input_ids):
        return type("Output", (), {"last_hidden_state": torch.ones((input_ids.shape[0], 2, 4))})()


def test_encode_options_jointly_pairs_question_with_each_candidate():
    row = {"question": "question", **{f"choice{i}": f"candidate {i}" for i in range(5)}}
    result = encode_options([row], _Tokenizer(), _Encoder(), batch_size=2, max_length=8)
    assert tuple(result.shape) == (1, 5, 4)


def test_choice_metrics_supports_five_options():
    logits = torch.tensor([[0.0, 0.0, 5.0, 0.0, 0.0], [4.0, 0.0, 0.0, 0.0, 0.0]])
    result = choice_metrics(logits, torch.tensor([2, 0]))
    assert result["count"] == 2
    assert result["accuracy"] == 1.0
    assert result["macro_accuracy_by_target_position"] == 1.0
    assert result["expected_random_accuracy"] == 0.2
