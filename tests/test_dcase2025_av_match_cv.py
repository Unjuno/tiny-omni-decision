import numpy as np
import pytest

torch = pytest.importorskip("torch")

from scripts.run_dcase2025_av_match_cv import _metrics, _shuffle_options, fit_fold  # noqa: E402


def test_four_choice_metrics_and_option_shuffle_are_valid():
    probabilities = np.full((4, 4), 0.25)
    targets = np.arange(4)
    metrics = _metrics(probabilities, targets)
    videos = np.arange(4 * 4 * 2, dtype=np.float32).reshape(4, 4, 2)

    shuffled = _shuffle_options(videos, [f"q{index}" for index in range(4)], seed=17)

    assert metrics["accuracy"] == 0.25
    assert metrics["nll"] == pytest.approx(np.log(4))
    assert metrics["brier"] == pytest.approx(0.75)
    assert all(not np.array_equal(videos[i], shuffled[i]) for i in range(4))


def test_rank_eight_match_fit_selects_on_validation_and_returns_small_model():
    rng = np.random.default_rng(17)
    audio_train = rng.normal(size=(48, 768)).astype(np.float32)
    video_train = rng.normal(size=(48, 4, 768)).astype(np.float32)
    audio_valid = rng.normal(size=(12, 768)).astype(np.float32)
    video_valid = rng.normal(size=(12, 4, 768)).astype(np.float32)
    target_train = np.arange(48, dtype=np.int64) % 4
    target_valid = np.arange(12, dtype=np.int64) % 4

    model, fit = fit_fold(
        audio_train,
        video_train,
        target_train,
        audio_valid,
        video_valid,
        target_valid,
        seed=17,
        rank=8,
        max_epochs=2,
        patience=1,
    )

    assert fit["trainable_parameters"] == 12_288
    assert 1 <= fit["best_epoch"] <= fit["epochs_run"] <= 2
    assert np.isfinite(fit["validation_metrics"]["nll"])
    assert sum(parameter.numel() for parameter in model.parameters()) == 12_288
