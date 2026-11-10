from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from scripts import videomae_count_feature_probe as probe


def _row(sample_id: str, *, count: int, asset: str | None = None) -> tuple[dict, dict]:
    media = {
        "id": sample_id,
        "asset_id": asset or sample_id,
        "archive_member": f"images/{sample_id}.png",
        "media_sha256": hashlib.sha256(sample_id.encode()).hexdigest(),
    }
    label = {"id": sample_id, "count": count}
    return media, label


def test_build_allowed_rows_preserves_only_train_and_selector_order() -> None:
    train = [_row("a", count=1), _row("b", count=2)]
    selector = [_row("c", count=3)]
    media, labels = probe.build_allowed_rows(
        {"train": [x[0] for x in train], "selector": [x[0] for x in selector]},
        {"train": [x[1] for x in train], "selector": [x[1] for x in selector]},
    )

    assert [row["id"] for row in media] == ["a", "b", "c"]
    assert [row["split"] for row in media] == ["train", "train", "selector"]
    assert [row["count"] for row in labels] == [1, 2, 3]
    assert all(
        set(row) == {"id", "asset_id", "archive_member", "media_sha256", "split"} for row in media
    )


@pytest.mark.parametrize("collision", ["id", "asset", "media"])
def test_build_allowed_rows_rejects_cross_split_identity_overlap(collision: str) -> None:
    train_media, train_label = _row("a", count=1)
    selector_media, selector_label = _row("b", count=2)
    if collision == "id":
        selector_media["id"] = selector_label["id"] = "a"
    elif collision == "asset":
        selector_media["asset_id"] = train_media["asset_id"]
    else:
        selector_media["media_sha256"] = train_media["media_sha256"]

    with pytest.raises(ValueError, match="overlap|ID order"):
        probe.build_allowed_rows(
            {"train": [train_media], "selector": [selector_media]},
            {"train": [train_label], "selector": [selector_label]},
        )


def test_prepare_reads_only_hash_pinned_train_and_selector_files(tmp_path, monkeypatch) -> None:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    media_train, label_train = _row("a", count=1)
    media_selector, label_selector = _row("b", count=2)
    files = {
        "train-media-manifest.jsonl": [media_train],
        "selector-media-manifest.jsonl": [media_selector],
        "train-labels.jsonl": [label_train],
        "selector-labels.jsonl": [label_selector],
    }
    file_hashes = {}
    for name, rows in files.items():
        path = inputs / name
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        file_hashes[name] = probe.sha256_file(path)
    train_order = probe._order_hash(["a"])
    selector_order = probe._order_hash(["b"])
    (inputs / "receipt.json").write_text(
        json.dumps(
            {
                "config_sha256": "test-config",
                "file_hashes": file_hashes,
                "train_id_order_sha256": train_order,
                "selector_id_order_sha256": selector_order,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(probe, "EXPECTED_COUNTS", {"train": 1, "selector": 1})
    monkeypatch.setattr(probe, "EXPECTED_INPUT_HASHES", file_hashes)
    monkeypatch.setattr(probe, "EXPECTED_CONFIG_SHA256", "test-config")

    output = tmp_path / "prepared"
    result = probe.prepare(inputs, output)

    assert result["report_pool_read"] is False
    assert result["official_validation_read"] is False
    assert result["sealed_audit_read"] is False
    assert {path.name for path in output.iterdir()} == {
        "media-manifest.jsonl",
        "labels.jsonl",
        "input-receipt.json",
    }
    assert [
        json.loads(line)["split"]
        for line in (output / "media-manifest.jsonl").read_text().splitlines()
    ] == [
        "train",
        "selector",
    ]


def test_fixed_ridge_probe_recovers_separable_count_classes() -> None:
    rng = np.random.default_rng(17)
    labels = np.repeat(np.arange(1, 11), 8)
    features = np.eye(10, dtype=np.float64)[labels - 1]
    features = np.pad(features, ((0, 0), (0, 3))) + rng.normal(0, 0.01, (80, 13))
    selector_labels = np.arange(1, 11)
    selector = np.pad(np.eye(10, dtype=np.float64), ((0, 0), (0, 3)))
    predictions, scores = probe.ridge_predict(features, labels, selector)

    assert predictions.tolist() == selector_labels.tolist()
    assert scores.shape == (10, 10)
    assert probe._summarize(selector_labels.tolist(), predictions)["accuracy"] == 1.0


def test_ridge_probe_rejects_unfrozen_hyperparameter_change() -> None:
    x = np.ones((10, 2), dtype=np.float32)
    y = np.arange(1, 11)
    with pytest.raises(ValueError, match="lambda"):
        probe.ridge_predict(x, y, x[:2], regularization=0.5)
