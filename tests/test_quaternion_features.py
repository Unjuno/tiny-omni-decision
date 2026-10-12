import numpy as np
import pytest

from tiny_omni_decision.quaternion_features import (
    quaternion_rotation_matrices,
    temporal_orientation_features,
)


def test_sign_invariance_and_known_rotation():
    q = np.array([[1., 0, 0, 0], [0, 0, 0, 1], [2**-.5, 0, 0, 2**-.5]])
    r = quaternion_rotation_matrices(q)
    np.testing.assert_allclose(r, quaternion_rotation_matrices(-q))
    np.testing.assert_allclose(r[0], np.eye(3))
    np.testing.assert_allclose(r[1], np.diag([-1, -1, 1]))
    np.testing.assert_allclose(r[2] @ [1, 0, 0], [0, 1, 0], atol=1e-14)
    np.testing.assert_allclose(r @ r.swapaxes(-1, -2), np.broadcast_to(np.eye(3), r.shape),
                               atol=1e-14)
    np.testing.assert_allclose(np.linalg.det(r), 1)


@pytest.mark.parametrize("q", [[[0, 0, 0, 0]], [[2, 0, 0, 0]], [[1, 0, 0]],
                             [[float("nan"), 0, 0, 0]]])
def test_invalid_quaternions_rejected(q):
    with pytest.raises(ValueError):
        quaternion_rotation_matrices(q)


def test_temporal_features_keep_bins_and_sign_invariance():
    q = np.zeros((16, 5, 4))
    q[:8, :, 0] = 1
    q[8:, :, 3] = 1
    t = np.arange(16) * 20
    features = temporal_orientation_features(t, q)
    assert features.shape == (8, 5, 3, 3)
    np.testing.assert_array_equal(features, temporal_orientation_features(t, -q))
    np.testing.assert_allclose(features[0, 0], np.eye(3))
    np.testing.assert_allclose(features[-1, 0], np.diag([-1, -1, 1]))
    with pytest.raises(ValueError):
        temporal_orientation_features([0, 0], q[:2])
    with pytest.raises(ValueError, match="empty"):
        temporal_orientation_features([0, 100], q[:2])
