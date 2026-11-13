import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
import torch
from transformers import BertConfig, BertForSequenceClassification

from scripts.train_text_crossencoder import choice_loss, configure_trainable, frozen_hash


def test_only_final_block_and_classifier_update():
    model = BertForSequenceClassification(
        BertConfig(
            hidden_size=8,
            num_hidden_layers=6,
            num_attention_heads=2,
            intermediate_size=16,
            vocab_size=30,
            num_labels=1,
        )
    )
    names = configure_trainable(model)
    assert names and all(n.startswith(("bert.encoder.layer.5.", "classifier.")) for n in names)
    before = frozen_hash(model)
    model.eval()
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
    loss = model(input_ids=torch.tensor([[1, 2, 3], [3, 2, 1]])).logits.sum()
    loss.backward()
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    optimizer.step()
    assert frozen_hash(model) == before


def test_variable_options_loss_weights_questions_equally():
    scores = torch.zeros(5, requires_grad=True)
    loss = choice_loss(scores, [2, 3], torch.tensor([0, 2]))
    expected = (torch.log(torch.tensor(2.0)) + torch.log(torch.tensor(3.0))) / 2
    assert torch.allclose(loss, expected)
    loss.backward()
    assert torch.isfinite(scores.grad).all()


def test_repair_encoder_policy_preserves_embeddings_and_pooler():
    from scripts.train_conditional_repair import configure_repair_policy

    model = BertForSequenceClassification(
        BertConfig(
            hidden_size=8,
            num_hidden_layers=6,
            num_attention_heads=2,
            intermediate_size=16,
            vocab_size=30,
            num_labels=1,
        )
    )
    legacy = configure_trainable(model)
    assert configure_repair_policy(model, "last_block") == legacy
    names = configure_repair_policy(model, "all_encoder_blocks")
    assert all(n.startswith(("bert.encoder.layer.", "classifier.")) for n in names)
    assert {n.split(".")[3] for n in names if n.startswith("bert.encoder.layer.")} == {
        str(i) for i in range(6)
    }
    before = frozen_hash(model)
    model.eval()
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
    model(input_ids=torch.tensor([[1, 2, 3], [3, 2, 1]])).logits.sum().backward()
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    optimizer.step()
    assert before == frozen_hash(model)
    with pytest.raises(ValueError):
        configure_repair_policy(model, "all_linear")
