from __future__ import annotations

import numpy as np
import pytest

from scripts.train_physical_concepts_next_feature_predictor import persistence_cosine_loss


def test_persistence_baseline_reports_cosine_distance() -> None:
    features = np.asarray([[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]], dtype=np.float32)
    assert persistence_cosine_loss(features) == pytest.approx(1.0)


def test_persistence_baseline_is_zero_for_identical_frame_features() -> None:
    features = np.ones((2, 3, 4), dtype=np.float32)
    assert persistence_cosine_loss(features) == pytest.approx(0.0)


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
