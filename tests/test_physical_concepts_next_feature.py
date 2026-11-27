from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from scripts.train_physical_concepts_next_feature_predictor import (
    evaluate_validation_loss,
    persistence_cosine_loss,
)
from tiny_omni_decision.temporal_prediction import NextFrameFeaturePredictor


def test_persistence_baseline_reports_cosine_distance() -> None:
    features = np.asarray([[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]], dtype=np.float32)
    assert persistence_cosine_loss(features) == pytest.approx(1.0)


def test_persistence_baseline_is_zero_for_identical_frame_features() -> None:
    features = np.ones((2, 3, 4), dtype=np.float32)
    assert persistence_cosine_loss(features) == pytest.approx(0.0)


def test_validation_evaluation_consumes_all_sequences() -> None:
    import torch

    torch.manual_seed(7)
    model = NextFrameFeaturePredictor(feature_width=4, hidden_width=2)
    features = np.random.default_rng(7).normal(size=(3, 4, 4)).astype(np.float32)
    assert 0.0 <= evaluate_validation_loss(model, features, batch_size=2) <= 2.0


def test_validation_evaluation_rejects_nonpositive_batch_size() -> None:
    model = NextFrameFeaturePredictor(feature_width=2, hidden_width=2)
    features = np.ones((1, 2, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="batch size must be positive"):
        evaluate_validation_loss(model, features, batch_size=0)


@pytest.mark.parametrize(
    "features",
    [
        np.ones((0, 3, 2), dtype=np.float32),
        np.ones((1, 1, 2), dtype=np.float32),
        np.full((1, 2, 2), np.nan, dtype=np.float32),
        np.asarray([[[0.0, 0.0], [1.0, 0.0]]], dtype=np.float32),
    ],
)
def test_persistence_baseline_rejects_invalid_features(features: np.ndarray) -> None:
    with pytest.raises(ValueError):
        persistence_cosine_loss(features)
