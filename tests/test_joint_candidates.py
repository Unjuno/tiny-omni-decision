import pytest

from tiny_omni_decision.joint_candidates import encode_joint


class Tokenizer:
    cls_token_id = 1
    sep_token_id = 2

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [3] * len(text.split())}


def test_joint_preserves_options_and_truncates_state_only():
    b, markers, meta = encode_joint(Tokenizer(), "pick minimum", "long " * 100,
                                     ["a", "b", "c"], max_length=24)
    assert len(b["input_ids"]) == 24
    assert len(markers) == 3
    assert all(b["input_ids"][p] == 2 for p in markers)
    assert meta["state_truncated"]
    assert len(b["token_type_ids"]) == len(b["input_ids"])
    with pytest.raises(ValueError, match="budget"):
        encode_joint(Tokenizer(), "long " * 40, "state", ["a", "b"], max_length=24)


def test_joint_reuses_weights_and_reaches_encoder_gradients():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from tiny_omni_decision.joint_candidates import joint_logits

    model = transformers.BertForSequenceClassification(transformers.BertConfig(
        vocab_size=12, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_labels=1,
    )).eval()
    count = sum(p.numel() for p in model.parameters())
    batch = {"input_ids": torch.tensor([[1, 3, 2, 4, 2, 5]]),
             "attention_mask": torch.ones(1, 6, dtype=torch.long)}
    scores = joint_logits(model, batch, [[2, 4]])[0]
    assert scores.shape == (2,)
    assert torch.isfinite(scores).all()
    scores.sum().backward()
    assert model.bert.embeddings.word_embeddings.weight.grad.abs().sum() > 0
    assert count == sum(p.numel() for p in model.parameters())
    with pytest.raises(ValueError, match="outside"):
        joint_logits(model, batch, [[2, 8]])
