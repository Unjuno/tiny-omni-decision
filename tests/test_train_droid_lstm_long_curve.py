from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
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
