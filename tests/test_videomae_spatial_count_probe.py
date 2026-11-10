import pytest

pytest.importorskip("torch")

from scripts import videomae_spatial_count_probe as probe


def _rows(prefix: str, count: int, amount: int) -> tuple[list[dict], list[dict]]:
    media, labels = [], []
    for i in range(amount):
        sample_id = f"{prefix}-{count}-{i}"
        media.append(
            {
                "id": sample_id,
                "asset_id": sample_id,
                "archive_member": f"images/{sample_id}.png",
                "media_sha256": f"hash-{sample_id}",
            }
        )
        labels.append({"id": sample_id, "count": count})
    return media, labels


def test_stable_split_is_stratified_deterministic_and_asset_disjoint(monkeypatch):
    monkeypatch.setattr(probe, "PER_CLASS", 8)
    monkeypatch.setattr(probe, "TRAIN_PER_CLASS", 6)
    all_rows = [_rows(f"source-{count}", count, 10) for count in range(1, 11)]
    media = [row for pair in all_rows for row in pair[0]]
    labels = [row for pair in all_rows for row in pair[1]]

    train_a, holdout_a = probe.stable_split(media, labels)
    train_b, holdout_b = probe.stable_split(media, labels)

    assert [r["id"] for r in train_a] == [r["id"] for r in train_b]
    assert [r["id"] for r in holdout_a] == [r["id"] for r in holdout_b]
    assert len(train_a) == 60
    assert len(holdout_a) == 20
    assert {r["asset_id"] for r in train_a}.isdisjoint({r["asset_id"] for r in holdout_a})
    assert {c: sum(r["count"] == c for r in train_a) for c in range(1, 11)} == {
        c: 6 for c in range(1, 11)
    }


def test_stable_split_rejects_duplicate_asset_identity(monkeypatch):
    monkeypatch.setattr(probe, "PER_CLASS", 1)
    monkeypatch.setattr(probe, "TRAIN_PER_CLASS", 1)
    pairs = [_rows(str(c), c, 1) for c in range(1, 11)]
    media = [row for pair in pairs for row in pair[0]]
    labels = [row for pair in pairs for row in pair[1]]
    media[1]["asset_id"] = media[0]["asset_id"]

    with pytest.raises(ValueError, match="duplicate train"):
        probe.stable_split(media, labels)


def test_metrics_report_perfect_multiclass_predictions():
    import numpy as np

    targets = np.arange(10)
    logits = np.eye(10, dtype=np.float32) * 8

    metrics = probe._metrics(targets, logits)

    assert metrics["accuracy"] == 1.0
    assert metrics["nll"] < 0.01
    assert metrics["brier"] < 0.001
    assert metrics["ece_15"] < 0.01


def test_spatial_encoder_retains_expected_token_and_grid_shapes():
    import torch

    class Stub:
        patch_embed = staticmethod(lambda x: torch.zeros(x.shape[0], 8 * 14 * 14, 384))
        pos_embed = None
        pos_drop = staticmethod(lambda x: x)
        blocks = []
        norm = staticmethod(lambda x: x)
        fc_norm = staticmethod(lambda x: x)

    pooled, token_mean, grid = probe._encode_spatial(Stub(), torch.zeros(2, 3, 16, 224, 224))

    assert pooled.shape == token_mean.shape == (2, 384)
    assert grid.shape == (2, 4 * 4 * 384)


def test_paired_bootstrap_is_deterministic():
    import numpy as np

    values = np.asarray([-1.0, 0.0, 1.0, 1.0])

    first = probe._paired_bootstrap(values, replicates=100, seed=17)
    second = probe._paired_bootstrap(values, replicates=100, seed=17)

    assert first == second
    assert first["estimate"] == 0.25
    assert first["unit"] == "one unique image/asset"


def test_summarize_predictions_keeps_paired_ids_and_class_metrics(tmp_path):
    import json

    predictions = tmp_path / "predictions.jsonl"
    rows = [
        {
            "id": f"asset-{i}",
            "target": i + 1,
            "global_logits": [8.0 if j == i else 0.0 for j in range(10)],
            "token_mean_logits": [8.0 if j == i else 0.0 for j in range(10)],
            "spatial_logits": [8.0 if j == i else 0.0 for j in range(10)],
        }
        for i in range(10)
    ]
    predictions.write_text("".join(json.dumps(row) + "\n" for row in rows))
    output = tmp_path / "statistics"

    result = probe.summarize_predictions(predictions, output)

    assert result["examples"] == result["unique_assets"] == 10
    assert result["metrics"]["global_384"]["accuracy"] == 1.0
    assert result["per_count_accuracy"]["spatial_grid_4x4"]["10"] == 1.0
    assert "correct_token_mean_384_minus_global_384" in result["paired_differences"]
    assert result["holdout_reused_for_model_selection"] is False
