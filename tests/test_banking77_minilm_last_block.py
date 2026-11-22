from types import SimpleNamespace

import torch
from torch import nn

from scripts.train_banking77_minilm_last_block import (
    collate_dynamic,
    configure_last_block,
    train_last_block,
)


class TinyBertLikeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = nn.Embedding(16, 8)
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList([nn.Linear(8, 8), nn.Linear(8, 8)])

    def forward(self, input_ids, attention_mask):
        hidden = self.embeddings(input_ids)
        for layer in self.encoder.layer:
            hidden = torch.tanh(layer(hidden))
        return SimpleNamespace(last_hidden_state=hidden)


def test_only_final_transformer_block_is_trainable():
    model = TinyBertLikeEncoder()
    trainable = configure_last_block(model)

    assert trainable
    assert all(parameter.requires_grad for parameter in model.encoder.layer[-1].parameters())
    assert all(not parameter.requires_grad for parameter in model.encoder.layer[0].parameters())
    assert all(not parameter.requires_grad for parameter in model.embeddings.parameters())


def test_dynamic_collate_pads_only_to_the_longest_example_in_batch():
    examples = [
        (
            {"input_ids": torch.tensor([4, 5]), "attention_mask": torch.tensor([1, 1])},
            torch.tensor(0),
        ),
        (
            {"input_ids": torch.tensor([6, 7, 8]), "attention_mask": torch.tensor([1, 1, 1])},
            torch.tensor(1),
        ),
    ]

    batch, labels = collate_dynamic(examples)

    assert batch["input_ids"].tolist() == [[4, 5, 0], [6, 7, 8]]
    assert batch["attention_mask"].tolist() == [[1, 1, 0], [1, 1, 1]]
    assert labels.tolist() == [0, 1]


def test_last_block_training_selects_a_finite_validation_checkpoint():
    torch.manual_seed(17)
    model = TinyBertLikeEncoder()
    train_ids = torch.randint(0, 16, (12, 6))
    train_mask = torch.ones_like(train_ids)
    val_ids = torch.randint(0, 16, (6, 6))
    val_mask = torch.ones_like(val_ids)
    train_labels = torch.tensor([0, 1] * 6)
    val_labels = torch.tensor([0, 1, 0, 1, 0, 1])
    train_tokens = [
        {"input_ids": ids, "attention_mask": mask}
        for ids, mask in zip(train_ids, train_mask, strict=True)
    ]
    val_tokens = [
        {"input_ids": ids, "attention_mask": mask}
        for ids, mask in zip(val_ids, val_mask, strict=True)
    ]
    head, fit = train_last_block(
        model,
        train_tokens,
        train_labels,
        val_tokens,
        val_labels,
        batch_size=4,
        max_epochs=3,
        patience=2,
    )

    assert 1 <= fit["best_epoch"] <= fit["epochs_completed"] <= 3
    assert torch.isfinite(torch.tensor(fit["best_validation_metrics"]["nll"]))
    assert head.out_features == 77
    assert sum(parameter.numel() for parameter in head.parameters()) == 693
    assert fit["trainable_encoder_parameters"] > 0
