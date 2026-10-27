from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import scripts.train_droid_dense_lstm_cv as reference  # noqa: E402
import scripts.train_droid_lstm_long_curve as runner  # noqa: E402


def test_milestone_scoring_does_not_consume_training_rng():
    torch.set_num_threads(1)
    torch.manual_seed(17)
    model = runner.SmallTemporalDecision()
    sensor = np.zeros((4, 8, 14), dtype=np.float32)
    history = np.eye(7, dtype=np.float32)[:4]
    state = torch.random.get_rng_state().clone()
    probabilities = runner.score(model, sensor, history, np.array([1, 3]))
    assert probabilities.shape == (2, 7)
    assert torch.equal(state, torch.random.get_rng_state())
    assert model.training


def test_short_mock_run_saves_milestone_without_optimizer(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "MAX_EPOCHS", 1)
    monkeypatch.setattr(runner, "MILESTONES", (1,))
    monkeypatch.setattr(runner.shutil, "disk_usage", lambda _: SimpleNamespace(free=10**12))
    torch.set_num_threads(1)
    sensor = np.zeros((4, 8, 14), dtype=np.float32)
    history = np.eye(7, dtype=np.float32)[:4]
    targets = np.array([0, 2, 2, 3])
    previous = np.array([0, 1, 2, 3])
    mean, scale = np.zeros(14, dtype=np.float32), np.ones(14, dtype=np.float32)
    results, curve, _, _ = runner.train_fold(
        sensor, history, targets, previous, np.array([0, 1]), np.array([2, 3]),
        0, tmp_path, None, mean, scale,
    )
    assert len(curve) == 1 and results[1][0].shape == (2, 7)
    with np.load(tmp_path / "held-fold-0-epoch-1.npz", allow_pickle=False) as file:
        assert file["held_indices"].tolist() == [2, 3]
    with np.load(tmp_path / "lstm-fold-0-epoch-1.npz", allow_pickle=False) as file:
        assert "temporal.weight_ih_l0" in file.files
        assert all("optimizer" not in key for key in file.files)


def test_epoch_50_matches_original_training_path(tmp_path, monkeypatch):
    monkeypatch.setattr(runner.shutil, "disk_usage", lambda _: SimpleNamespace(free=10**12))
    monkeypatch.setattr(runner, "MAX_EPOCHS", 50)
    monkeypatch.setattr(runner, "MILESTONES", (50,))
    torch.set_num_threads(1)
    sensor = np.random.default_rng(23).normal(size=(8, 8, 14)).astype(np.float32)
    previous = np.arange(8, dtype=np.int64) % 7
    targets = previous.copy()
    targets[[1, 3, 6]] = (targets[[1, 3, 6]] + 1) % 7
    history = np.eye(7, dtype=np.float32)[previous]
    train, held = np.arange(6), np.arange(6, 8)
    model, expected_curve, _, _ = reference.fit_fold(
        sensor, history, targets, previous, train, 0, tmp_path
    )
    expected = runner.score(model, sensor, history, held)
    results, curve, _, _ = runner.train_fold(
        sensor, history, targets, previous, train, held, 0, tmp_path, expected,
        np.zeros(14, dtype=np.float32), np.ones(14, dtype=np.float32),
    )
    np.testing.assert_array_equal(results[50][0], expected)
    np.testing.assert_array_equal(curve, expected_curve)
