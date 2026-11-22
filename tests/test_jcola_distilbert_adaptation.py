from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from fit_jcola_distilbert_adaptation import (  # noqa: E402
    configure_upper_layers,
    source_group_folds,
)


class _DistilLike(torch.nn.Module):
    def __init__(self, width: int, layers: int) -> None:
        super().__init__()
        self.embeddings = torch.nn.Linear(width, width)
        self.transformer = torch.nn.Module()
        self.transformer.layer = torch.nn.ModuleList(
            [torch.nn.Linear(width, width) for _ in range(layers)]
        )


def _rows() -> list[dict[str, object]]:
    return [
        {
            "uid": f"{source}-{label}-{index}",
            "source": source,
            "label": label,
            "sentence": f"{source} sentence {label} {index}",
        }
        for source in ("s1", "s2", "s3", "s4")
        for label in (0, 1)
        for index in range(3)
    ]


def test_source_group_folds_are_deterministic_complete_and_disjoint() -> None:
    rows = _rows()
    folds = source_group_folds(rows, seed=17)
    repeated = source_group_folds(rows, seed=17)
    assert [fold[2] for fold in folds] == [fold[2] for fold in repeated]
    assert len(folds) == 4
    held_out = []
    for train, validation, source in folds:
        train_sources = {rows[index]["source"] for index in train}
        validation_sources = {rows[index]["source"] for index in validation}
        assert train_sources.isdisjoint(validation_sources)
        assert validation_sources == {source}
        assert len(validation) == 6
        held_out.extend(validation)
    assert sorted(held_out) == list(range(len(rows)))


def test_upper_layer_adaptation_freezes_embeddings_and_lower_layers() -> None:
    model = _DistilLike(width=3, layers=4)
    count = configure_upper_layers(model, train_last_layers=2)
    assert count == sum(
        p.numel() for layer in model.transformer.layer[2:] for p in layer.parameters()
    )
    assert not any(p.requires_grad for p in model.embeddings.parameters())
    assert not any(
        p.requires_grad for layer in model.transformer.layer[:2] for p in layer.parameters()
    )
    assert all(p.requires_grad for layer in model.transformer.layer[2:] for p in layer.parameters())


def test_invalid_layer_subset_fails_closed() -> None:
    model = _DistilLike(width=2, layers=4)
    with pytest.raises(ValueError, match="proper upper-layer subset"):
        configure_upper_layers(model, train_last_layers=4)
