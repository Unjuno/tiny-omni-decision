import pytest

np = pytest.importorskip("numpy")

from tiny_omni_decision.trajectory_features import trajectory_features  # noqa: E402


def test_fixed_temporal_order_and_all_points():
    tracks = np.arange(2 * 5 * 2, dtype=np.float32).reshape(2, 5, 2)
    logits = np.zeros((2, 5), dtype=np.float32)
    features, indices = trajectory_features(
        tracks, logits, logits, width=256, height=256, samples=3
    )
    assert indices.tolist() == [0, 2, 4]
    assert features.shape == (2, 12)
    np.testing.assert_allclose(features.reshape(2, 3, 4)[..., :2], tracks[:, indices] / 256)
    np.testing.assert_allclose(features.reshape(2, 3, 4)[..., 2:], 0.5)


def test_reverse_and_permutation_preserved():
    tracks = np.arange(20, dtype=np.float32).reshape(2, 5, 2)
    logits = np.zeros((2, 5), dtype=np.float32)
    normal, _ = trajectory_features(tracks, logits, logits, samples=5)
    reverse, _ = trajectory_features(tracks[:, ::-1], logits, logits, samples=5)
    permuted, _ = trajectory_features(tracks[::-1], logits, logits, samples=5)
    assert not np.array_equal(normal, reverse)
    np.testing.assert_array_equal(normal[::-1], permuted)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_nonfinite_rejected(bad):
    tracks = np.zeros((2, 5, 2), dtype=np.float32)
    tracks[0, 0, 0] = bad
    with pytest.raises(ValueError):
        trajectory_features(tracks, np.zeros((2, 5)), np.zeros((2, 5)))


def test_shape_and_empty_rejected():
    with pytest.raises(ValueError):
        trajectory_features(np.zeros((0, 5, 2)), np.zeros((0, 5)), np.zeros((0, 5)))
    with pytest.raises(ValueError):
        trajectory_features(np.zeros((2, 5, 2)), np.zeros((1, 5)), np.zeros((2, 5)))

