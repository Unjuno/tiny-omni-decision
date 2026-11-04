import numpy as np
import pytest

torch = pytest.importorskip("torch")

from scripts.run_dcase2025_nested_room_linear_cv import fit_fold_mode  # noqa: E402


@pytest.mark.parametrize(
    "mode,expected_parameters", [("audio", 1564), ("video", 1564), ("joint", 3100)]
)
def test_linear_nested_fold_fit_selects_by_validation_without_nonfinite_values(
    mode, expected_parameters
):
    rng = np.random.default_rng(17)
    train_audio = rng.normal(size=(48, 768)).astype(np.float32)
    train_video = rng.normal(size=(48, 768)).astype(np.float32)
    valid_audio = rng.normal(size=(12, 768)).astype(np.float32)
    valid_video = rng.normal(size=(12, 768)).astype(np.float32)
    train_events = np.arange(48, dtype=np.int64) % 13
    valid_events = np.arange(12, dtype=np.int64) % 13
    train_targets = train_events % 2
    valid_targets = valid_events % 2

    model, fit = fit_fold_mode(
        mode,
        train_audio,
        train_video,
        train_events,
        train_targets,
        valid_audio,
        valid_video,
        valid_events,
        valid_targets,
        seed=17,
        max_epochs=3,
        patience=2,
    )

    assert fit["trainable_parameters"] == expected_parameters
    assert 1 <= fit["best_epoch"] <= fit["epochs_run"] <= 3
    assert np.isfinite(fit["selected_validation_metrics"]["nll"])
    assert sum(parameter.numel() for parameter in model.parameters()) == expected_parameters
