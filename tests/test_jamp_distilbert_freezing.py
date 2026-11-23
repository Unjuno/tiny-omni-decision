from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from scripts.pretrained_reuse.finetune_jamp_distilbert_top_layers import (  # noqa: E402
    freeze_except_top_layers,
)


class _FakeDistilBert(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(model_type="distilbert", n_layers=4)
        self.embeddings = torch.nn.Embedding(8, 4)
        self.transformer = torch.nn.Module()
        self.transformer.layer = torch.nn.ModuleList(
            [torch.nn.Linear(4, 4) for _ in range(self.config.n_layers)]
        )


def test_distilbert_freezes_embeddings_and_lower_layers() -> None:
    model = _FakeDistilBert()

    trainable = freeze_except_top_layers(model, trainable_layers=2)

    assert trainable
    assert all(
        name.startswith(("transformer.layer.2.", "transformer.layer.3.")) for name in trainable
    )
    assert all(not parameter.requires_grad for parameter in model.embeddings.parameters())
    assert all(not parameter.requires_grad for parameter in model.transformer.layer[0].parameters())
    assert all(not parameter.requires_grad for parameter in model.transformer.layer[1].parameters())
    assert all(parameter.requires_grad for parameter in model.transformer.layer[2].parameters())
    assert all(parameter.requires_grad for parameter in model.transformer.layer[3].parameters())


def test_distilbert_freeze_rejects_unfreezing_every_layer() -> None:
    with pytest.raises(ValueError, match="trainable_layers"):
        freeze_except_top_layers(_FakeDistilBert(), trainable_layers=4)
