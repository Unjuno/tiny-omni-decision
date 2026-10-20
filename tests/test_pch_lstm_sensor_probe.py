import numpy as np
import pytest

from scripts.pch_lstm_sensor_probe import _build_model, _record_logits, _window_arrays
from scripts.pch_moment_sensor_probe import ACTIVITY_IDS


def test_pch_lstm_consumes_temporal_axis_and_produces_finite_gradients():
    torch = pytest.importorskip("torch")
    torch.manual_seed(17)
    model = _build_model(torch, hidden_size=8, num_layers=1)
    windows = torch.randn(4, 128, 3)

    logits = model(windows)
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 2, 3]))
    loss.backward()

    assert logits.shape == (4, len(ACTIVITY_IDS))
    assert torch.isfinite(logits).all()
    assert model.lstm.weight_ih_l0.grad is not None
    assert torch.isfinite(model.lstm.weight_ih_l0.grad).all()
    assert torch.count_nonzero(model.lstm.weight_ih_l0.grad) > 0


def test_window_flattening_preserves_record_labels_and_record_ownership():
    records = [
        {
            "sample_id": "a",
            "subject_id": "P001",
            "label": ACTIVITY_IDS[0],
            "windows": np.zeros((2, 3, 128), np.float32),
            "window_count": 2,
        },
        {
            "sample_id": "b",
            "subject_id": "P002",
            "label": ACTIVITY_IDS[1],
            "windows": np.ones((1, 3, 128), np.float32),
            "window_count": 1,
        },
    ]
    split = {"fit_subjects": ["P001"], "validation_subjects": ["P002"]}

    result = _window_arrays(records, split)

    fit_windows, fit_targets, fit_owners, fit_counts = result["fit"]
    val_windows, val_targets, val_owners, val_counts = result["validation"]
    assert fit_windows.shape == (2, 3, 128)
    assert fit_targets.tolist() == [0, 0]
    assert fit_owners.tolist() == [0, 0]
    assert fit_counts.tolist() == [2]
    assert val_windows.shape == (1, 3, 128)
    assert val_targets.tolist() == [1]
    assert val_owners.tolist() == [0]
    assert val_counts.tolist() == [1]


def test_record_logits_mean_pools_window_logits_by_record():
    torch = pytest.importorskip("torch")

    class MeanSignalModel(torch.nn.Module):
        def forward(self, rows):
            scores = rows.mean(dim=(1, 2))
            return torch.stack((scores, -scores), dim=-1)

    windows = np.zeros((3, 3, 128), dtype=np.float32)
    windows[1] = 2.0
    windows[2] = -1.0
    owners = np.asarray([0, 0, 1], dtype=np.int64)
    mean = torch.zeros((1, 3, 1))
    scale = torch.ones((1, 3, 1))

    result = _record_logits(
        torch,
        MeanSignalModel(),
        windows,
        owners,
        record_count=2,
        mean=mean,
        scale=scale,
        device=torch.device("cpu"),
        batch_size=2,
    )

    assert result.shape == (2, 2)
    assert result[0, 0] == pytest.approx(1.0)
    assert result[1, 0] == pytest.approx(-1.0)
