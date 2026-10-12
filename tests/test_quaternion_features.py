import numpy as np
import pytest

from tiny_omni_decision.quaternion_features import quaternion_rotation_matrices


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
