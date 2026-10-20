import hashlib
import importlib
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
_runner = importlib.import_module("scripts.train_av_event_source")
fit_readout = _runner.fit_readout
load_feature_rows = _runner.load_feature_rows
pool_dinov2_patch_tokens = _runner.pool_dinov2_patch_tokens
select_best_epoch = _runner.select_best_epoch
metric_summary = _runner._metric_summary
validate_video_splits = _runner.validate_video_splits


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_dinov2_pooling_uses_class_token_without_mixing_candidates():
    tokens = np.zeros((2, 257, 384), dtype=np.float32)
    tokens[0, 0] = 3.0
    tokens[0, 1:] = 99.0
    tokens[1, 0] = -2.0
    tokens[1, 1:] = 101.0
    pooled = pool_dinov2_patch_tokens(tokens)
    assert pooled.shape == (2, 384)
    assert np.array_equal(pooled[:, 0], np.array([3.0, -2.0], dtype=np.float32))
    assert np.array_equal(pool_dinov2_patch_tokens(tokens[:, 0, :]), pooled)


def test_video_split_validation_rejects_same_scene_in_fit_and_selection():
    rows = [
        {"id": "fit-1", "video_id": "video_1", "split": "fit"},
        {"id": "selection-1", "video_id": "video_1", "split": "selection"},
    ]
    with pytest.raises(ValueError, match="video appears in multiple splits"):
        validate_video_splits(rows)


def test_video_split_validation_rejects_media_hash_reuse_across_splits():
    rows = [
        {"id": "fit-1", "video_id": "video_1", "split": "fit", "media_sha256": "a" * 64},
        {
            "id": "selection-1",
            "video_id": "video_2",
            "split": "selection",
            "media_sha256": "a" * 64,
        },
    ]
    with pytest.raises(ValueError, match="media identity appears in multiple splits"):
        validate_video_splits(rows)


def test_feature_loader_rejects_label_out_of_candidate_range(tmp_path):
    audio = tmp_path / "audio.npy"
    visual = tmp_path / "visual.npy"
    np.save(audio, np.zeros((512,), dtype=np.float32))
    np.save(visual, np.zeros((3, 257, 384), dtype=np.float32))
    rows = [
        {
            "id": "event-1",
            "video_id": "video_1",
            "split": "fit",
            "target_index": 3,
            "candidate_count": 3,
            "audio_feature_path": str(audio),
            "audio_feature_sha256": _sha256(audio),
            "visual_feature_path": str(visual),
            "visual_feature_sha256": _sha256(visual),
            "candidate_order_sha256": "c" * 64,
            "media_sha256": "a" * 64,
            "annotation_sha256": "b" * 64,
        }
    ]
    manifest = tmp_path / "rows.jsonl"
    manifest.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="target index outside candidate list"):
        load_feature_rows(manifest)


def test_feature_loader_requires_candidate_order_digest(tmp_path):
    audio = tmp_path / "audio.npy"
    visual = tmp_path / "visual.npy"
    np.save(audio, np.zeros((512,), dtype=np.float32))
    np.save(visual, np.zeros((2, 257, 384), dtype=np.float32))
    row = {
        "id": "event-1",
        "video_id": "video_1",
        "split": "fit",
        "target_index": 0,
        "candidate_count": 2,
        "audio_feature_path": str(audio),
        "audio_feature_sha256": _sha256(audio),
        "visual_feature_path": str(visual),
        "visual_feature_sha256": _sha256(visual),
        "media_sha256": "a" * 64,
        "annotation_sha256": "b" * 64,
    }
    manifest = tmp_path / "rows.jsonl"
    manifest.write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises(ValueError, match="candidate_order_sha256"):
        load_feature_rows(manifest)


def test_best_epoch_uses_selection_nll_and_keeps_earliest_tie():
    history = [
        {"epoch": 1, "selection_nll": 0.8},
        {"epoch": 2, "selection_nll": 0.6},
        {"epoch": 3, "selection_nll": 0.6},
        {"epoch": 4, "selection_nll": 0.7},
    ]
    assert select_best_epoch(history) == 2


def test_best_epoch_applies_absolute_min_delta_like_checkpoint_selector():
    history = [
        {"epoch": 1, "selection_nll": 0.50},
        {"epoch": 2, "selection_nll": 0.45},
        {"epoch": 3, "selection_nll": 0.40},
    ]
    assert select_best_epoch(history, min_delta=0.10) == 1


def test_metric_summary_supports_events_with_different_candidate_counts():
    logits = [torch.tensor([3.0, 0.0]), torch.tensor([0.0, 3.0, 1.0])]
    metrics = metric_summary(logits, [0, 1])
    assert metrics["accuracy"] == 1.0
    assert np.isfinite(metrics["nll"])
    assert np.isfinite(metrics["brier"])
    assert np.isfinite(metrics["ece"])


def test_fit_readout_requires_calibration_and_development_validation_splits(tmp_path):
    rows = [
        {"id": "fit", "video_id": "video_fit", "split": "fit"},
        {"id": "selection", "video_id": "video_selection", "split": "selection"},
    ]
    with pytest.raises(ValueError, match="calibration and development-validation"):
        fit_readout(
            rows,
            tmp_path / "run",
            condition="video_only",
            epochs=1,
            patience=1,
            hidden_size=4,
        )


def test_fit_readout_keeps_best_and_final_checkpoints_separate(tmp_path):
    rows = [
        {
            "id": "fit-a",
            "video_id": "video_fit_a",
            "split": "fit",
            "target_index": 0,
            "candidate_count": 2,
            "candidate_order_sha256": "c" * 64,
            "audio": np.zeros(512, dtype=np.float32),
            "visual": np.zeros((2, 384), dtype=np.float32),
        },
        {
            "id": "fit-b",
            "video_id": "video_fit_b",
            "split": "fit",
            "target_index": 1,
            "candidate_count": 2,
            "candidate_order_sha256": "d" * 64,
            "audio": np.ones(512, dtype=np.float32),
            "visual": np.ones((2, 384), dtype=np.float32),
        },
        {
            "id": "selection-a",
            "video_id": "video_selection_a",
            "split": "selection",
            "target_index": 1,
            "candidate_count": 2,
            "candidate_order_sha256": "e" * 64,
            "audio": np.full(512, 0.5, dtype=np.float32),
            "visual": np.full((2, 384), 0.5, dtype=np.float32),
        },
        {
            "id": "valid-a",
            "video_id": "video_validation_a",
            "split": "development_validation",
            "target_index": 0,
            "candidate_count": 2,
            "candidate_order_sha256": "f" * 64,
            "audio": np.full(512, -0.5, dtype=np.float32),
            "visual": np.full((2, 384), -0.5, dtype=np.float32),
        },
        {
            "id": "cal-a",
            "video_id": "video_calibration_a",
            "split": "calibration",
            "target_index": 0,
            "candidate_count": 2,
            "candidate_order_sha256": "1" * 64,
            "audio": np.full(512, -1.0, dtype=np.float32),
            "visual": np.full((2, 384), -1.0, dtype=np.float32),
        },
    ]
    output = tmp_path / "run"
    result = fit_readout(
        rows,
        output,
        condition="video_only",
        seed=17,
        epochs=1,
        patience=1,
        hidden_size=4,
    )
    assert result["status"] == "PASS_AV_EVENT_READOUT_FIT"
    assert result["best_epoch"] == 1
    assert result["trainable_parameters"] == 1565
    assert (output / "best.pt").is_file()
    assert (output / "final.pt").is_file()
    final_checkpoint = torch.load(output / "final.pt", map_location="cpu", weights_only=True)
    assert final_checkpoint["hidden_size"] == 4
    assert final_checkpoint["epoch"] == 1
    assert (output / "history.json").is_file()
    assert (output / "best-selection-predictions.jsonl").is_file()
    assert (output / "development-validation-predictions.jsonl").is_file()
    assert result["checkpoint_reload_verified"] is True
    assert result["development_validation_metrics"]["nll"] >= 0


def test_fit_readout_honors_fixed_update_budget_and_validation_interval(tmp_path):
    rows = []
    for split, suffix, target in (
        ("fit", "fit", 0),
        ("selection", "selection", 1),
        ("calibration", "calibration", 0),
        ("development_validation", "validation", 1),
    ):
        rows.append(
            {
                "id": suffix,
                "video_id": f"video_{suffix}",
                "split": split,
                "target_index": target,
                "candidate_count": 2,
                "candidate_order_sha256": "a" * 64,
                "audio": np.full(512, float(target), dtype=np.float32),
                "visual": np.full((2, 384), float(target), dtype=np.float32),
            }
        )
    output = tmp_path / "fixed-run"
    result = fit_readout(
        rows,
        output,
        condition="video_only",
        seed=17,
        epochs=10,
        patience=4,
        hidden_size=4,
        max_updates=6,
        evaluation_interval=2,
    )
    history = json.loads((output / "history.json").read_text(encoding="utf-8"))
    final = torch.load(output / "final.pt", map_location="cpu", weights_only=True)
    assert result["optimizer_updates"] == 6
    assert result["best_optimizer_updates"] in {2, 4, 6}
    assert result["evaluation_interval"] == 2
    assert [row["optimizer_updates"] for row in history] == [2, 4, 6]
    assert final["optimizer_updates"] == 6
    assert result["checkpoint_reload_verified"] is True


def test_fit_readout_requires_fixed_updates_and_interval_together(tmp_path):
    with pytest.raises(ValueError, match="must be supplied together"):
        fit_readout(
            [],
            tmp_path / "run",
            condition="joint",
            max_updates=10,
        )
