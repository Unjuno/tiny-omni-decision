import pytest

np = pytest.importorskip("numpy")


def test_identical_reference_and_deterministic_clusters():
    from tiny_omni_decision.paired_video_intervals import video_cluster_intervals

    p = np.array([[[0.9, 0.1], [0.1, 0.9]], [[0.1, 0.9], [0.9, 0.1]]])
    y = np.array([[0, 1], [0, 1]])
    result = video_cluster_intervals(p, y, reference=p, draws=100)
    assert result == video_cluster_intervals(p, y, reference=p, draws=100)
    assert all(value == [0.0, 0.0] for value in result["paired_difference_intervals"].values())
    # Both conditions agree within each video: clustering preserves this equality.
    assert result["intervals"]["accuracy"] == result["intervals"]["both_correct"]


def test_input_validation():
    from tiny_omni_decision.paired_video_intervals import video_cluster_intervals

    with pytest.raises(ValueError, match="sum to one"):
        video_cluster_intervals(np.ones((2, 2, 3)), np.zeros((2, 2), dtype=int))
    with pytest.raises(ValueError, match="aligned reference"):
        video_cluster_intervals(
            np.ones((2, 2, 3)) / 3, np.zeros((2, 2), dtype=int), reference=np.ones((1, 2, 3)) / 3
        )
