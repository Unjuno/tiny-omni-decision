from __future__ import annotations

import importlib
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

diagnostic = importlib.import_module("scripts.train_videomae_clevr4_count_linear_diagnostic")


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_count_targets_uses_fixed_taxonomy_order() -> None:
    ids, targets = diagnostic.count_targets(
        [{"id": "image-b", "count": "2"}, {"id": "image-a", "count": "10"}]
    )

    assert ids == ["image-b", "image-a"]
    assert targets.tolist() == [1, 9]


def test_count_targets_rejects_unknown_label_and_duplicate_id() -> None:
    with pytest.raises(ValueError, match="unknown count label"):
        diagnostic.count_targets([{"id": "x", "count": "11"}])
    with pytest.raises(ValueError, match="duplicate image IDs"):
        diagnostic.count_targets([{"id": "x", "count": "2"}, {"id": "x", "count": "3"}])


def test_count_linear_readout_has_expected_shape_and_size() -> None:
    model = diagnostic.CountLinearReadout()
    output = model(torch.zeros((3, diagnostic.FEATURE_WIDTH)))

    assert output.shape == (3, 10)
    assert sum(parameter.numel() for parameter in model.parameters()) == 3_850


def test_split_feature_loader_selects_only_frozen_train_and_selector_ids(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(diagnostic, "TRAIN_COUNT", 2)
    monkeypatch.setattr(diagnostic, "SELECTOR_COUNT", 1)
    monkeypatch.setattr(diagnostic, "FEATURE_COUNT", 3)
    rows = [
        {"id": "train-1", "media_sha256": "hash-1"},
        {"id": "train-2", "media_sha256": "hash-2"},
        {"id": "selector-1", "media_sha256": "hash-3"},
    ]
    manifest = tmp_path / "features.jsonl"
    _write_jsonl(manifest, rows)
    features = tmp_path / "features.npz"
    np.savez_compressed(
        features,
        ids=np.array([row["id"] for row in rows]),
        media_sha256=np.array([row["media_sha256"] for row in rows]),
        features=np.arange(3 * diagnostic.FEATURE_WIDTH, dtype=np.float32).reshape(3, -1),
    )

    train, selector, integrity = diagnostic._load_split_features(
        features_path=features,
        feature_manifest_path=manifest,
        train_ids=["train-1", "train-2"],
        selector_ids=["selector-1"],
        train_manifest=rows[:2],
        selector_manifest=rows[2:],
    )

    assert train.shape == (2, diagnostic.FEATURE_WIDTH)
    assert selector.shape == (1, diagnostic.FEATURE_WIDTH)
    expected = np.arange(
        2 * diagnostic.FEATURE_WIDTH,
        3 * diagnostic.FEATURE_WIDTH,
        dtype=np.float32,
    )
    assert torch.equal(selector[0], torch.from_numpy(expected))
    assert integrity["train_unique_media"] == 2
    assert integrity["selector_unique_media"] == 1


def test_split_feature_loader_rejects_media_overlap(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(diagnostic, "TRAIN_COUNT", 1)
    monkeypatch.setattr(diagnostic, "SELECTOR_COUNT", 1)
    with pytest.raises(ValueError, match="train and selector media identities overlap"):
        diagnostic._load_split_features(
            features_path=tmp_path / "unused.npz",
            feature_manifest_path=tmp_path / "unused.jsonl",
            train_ids=["train"],
            selector_ids=["selector"],
            train_manifest=[{"id": "train", "media_sha256": "same"}],
            selector_manifest=[{"id": "selector", "media_sha256": "same"}],
        )
