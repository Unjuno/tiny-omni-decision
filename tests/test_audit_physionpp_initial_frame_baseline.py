import numpy as np
import pytest

from scripts.audit_physionpp_initial_frame_baseline import (
    gaussian_nb_probabilities,
    spatial_rgb_features,
)


def test_spatial_rgb_features_are_fixed_grid_means():
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    frame[:8, :8] = (255, 0, 0)
    frame[:8, 8:] = (0, 255, 0)
    frame[8:, :8] = (0, 0, 255)
    frame[8:, 8:] = (255, 255, 255)

    result = spatial_rgb_features(frame, grid_size=2)

    assert result.shape == (12,)
    assert np.allclose(result, frame.reshape(2, 8, 2, 8, 3).mean(axis=(1, 3)).ravel() / 255)


def test_spatial_rgb_features_reject_bad_shapes():
    with pytest.raises(ValueError, match="HxWx3"):
        spatial_rgb_features(np.zeros((16, 16), dtype=np.uint8))
    with pytest.raises(ValueError, match="divisible"):
        spatial_rgb_features(np.zeros((15, 16, 3), dtype=np.uint8), grid_size=2)


def test_gaussian_naive_bayes_returns_finite_ordered_probabilities():
    train_x = np.asarray([[0.0, 0.1], [0.1, 0.0], [0.9, 1.0], [1.0, 0.9]])
    train_y = np.asarray([0, 0, 1, 1])
    test_x = np.asarray([[0.05, 0.05], [0.95, 0.95]])

    probabilities = gaussian_nb_probabilities(train_x, train_y, test_x)

    assert probabilities.shape == (2,)
    assert np.isfinite(probabilities).all()
    assert probabilities[0] < 0.5 < probabilities[1]


def test_gaussian_naive_bayes_rejects_single_class_training_fold():
    with pytest.raises(ValueError, match="both classes"):
        gaussian_nb_probabilities(np.zeros((2, 3)), np.zeros(2), np.zeros((1, 3)))
