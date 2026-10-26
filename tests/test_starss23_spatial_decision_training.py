import json

import numpy as np
import pytest
import torch

from scripts.extract_frozen_videoprism_features import sha256_file
from scripts.train_starss23_spatial_decision import (
    _feature_matrix,
    _fit_arm,
    balanced_cross_entropy,
    candidate_features,
    same_room_video_permutation,
    standardize_by_train_windows,
)


def test_candidate_features_encode_context_and_binary_option_order():
    observations = torch.tensor([[2.0, 3.0], [5.0, 7.0]])
    event_ids = torch.tensor([0, 11])
    sector_ids = torch.tensor([0, 7])

    values = candidate_features(observations, event_ids, sector_ids)

    assert values.shape == (2, 2, 2 + 12 + 8 + 2)
    torch.testing.assert_close(values[:, 0, :2], observations)
    torch.testing.assert_close(values[:, 0, 2:14].argmax(-1), event_ids)
    torch.testing.assert_close(values[:, 0, 14:22].argmax(-1), sector_ids)
    torch.testing.assert_close(values[:, :, -2:], torch.eye(2).expand(2, -1, -1))


def test_balanced_cross_entropy_weights_positive_and_negative_groups_equally():
    logits = torch.tensor([[4.0, 0.0], [4.0, 0.0], [0.0, 4.0]])
    targets = torch.tensor([0, 1, 1])
    losses = torch.nn.functional.cross_entropy(logits, targets, reduction="none")

    actual = balanced_cross_entropy(logits, targets)

    expected = (losses[targets == 0].mean() + losses[targets == 1].mean()) / 2
    torch.testing.assert_close(actual, expected)


def test_balanced_cross_entropy_handles_batch_without_positive_rows():
    logits = torch.tensor([[1.0, 0.0], [2.0, 0.0]])
    targets = torch.tensor([0, 0])

    actual = balanced_cross_entropy(logits, targets)

    torch.testing.assert_close(actual, torch.nn.functional.cross_entropy(logits, targets))


def test_feature_loader_rejects_feature_rows_with_wrong_media_identity(tmp_path):
    matrix = np.zeros((1, 1, 2), dtype=np.float32)
    path = tmp_path / "features.npy"
    np.save(path, matrix)
    feature = {
        "id": "window-1",
        "recording": "recording-1",
        "split": "train",
        "media_sha256": "wrong-media-hash",
    }
    report = {
        "status": "PASS",
        "modality": "audio",
        "manifest_sha256": "manifest-hash",
        "feature_sha256": sha256_file(path),
        "feature_shape": [1, 1, 2],
        "features": [feature],
    }
    (tmp_path / "result.json").write_text(json.dumps(report), encoding="utf-8")
    manifest = [
        {
            "id": "window-1",
            "recording": "recording-1",
            "split": "train",
            "media": {"audio": {"sha256": "correct-media-hash"}},
        }
    ]

    with pytest.raises(ValueError, match="media identity"):
        _feature_matrix(tmp_path, "audio", "manifest-hash", (1, 1, 2), manifest)


def test_feature_standardization_uses_only_unique_training_windows():
    features = np.asarray([[0.0, 10.0], [2.0, 14.0], [1000.0, -500.0]])

    standardized, mean, std = standardize_by_train_windows(features, np.asarray([0, 1, 1]))

    np.testing.assert_allclose(mean, [1.0, 12.0])
    np.testing.assert_allclose(standardized[:2].mean(axis=0), [0.0, 0.0], atol=1e-6)
    assert np.max(np.abs(standardized[2])) > 100
    assert np.isfinite(std).all()


def test_video_permutation_changes_recording_but_preserves_room():
    windows = np.arange(8)
    recordings = ["a", "a", "b", "b", "c", "c", "d", "d"]
    rooms = ["r1"] * 4 + ["r2"] * 4

    replacement = same_room_video_permutation(windows, recordings, rooms, seed=17)

    lookup = {int(window): i for i, window in enumerate(windows)}
    assert all(
        recordings[i] != recordings[lookup[int(target)]] for i, target in enumerate(replacement)
    )
    assert all(rooms[i] == rooms[lookup[int(target)]] for i, target in enumerate(replacement))


def test_cpu_training_smoke_saves_and_reloads_selected_checkpoint(tmp_path, monkeypatch):
    import scripts.train_starss23_spatial_decision as training

    monkeypatch.setattr(training, "MAX_EPOCHS", 1)
    monkeypatch.setattr(training, "BATCH_SIZE", 32)

    def rows(count):
        return {
            "ids": np.asarray([f"q-{i}" for i in range(count)]),
            "window": np.arange(count, dtype=np.int64),
            "event": np.tile(np.repeat(np.arange(12), 8), count // 96),
            "sector": np.tile(np.tile(np.arange(8), 12), count // 96),
            "target": np.arange(count, dtype=np.int64) % 2,
            "recording": np.asarray([f"recording-{i}" for i in range(count)]),
            "room": np.asarray([f"room-{i}" for i in range(count)]),
        }

    splits = {"train": rows(192), "calibration": rows(96), "validation": rows(96)}
    features = np.random.default_rng(17).normal(size=(192, 4)).astype(np.float32)
    arm_dir = tmp_path / "audio_only"
    arm_dir.mkdir()

    result = _fit_arm(
        "audio_only",
        features,
        splits,
        {},
        arm_dir,
        torch.device("cpu"),
        (np.zeros(4, dtype=np.float32), np.ones(4, dtype=np.float32)),
    )

    assert result["best_epoch"] == 1
    assert (arm_dir / "best.pt").is_file()
    assert (arm_dir / "final.pt").is_file()
    assert (arm_dir / "validation_predictions.npz").is_file()
