import numpy as np
import pytest

from tiny_omni_decision.cluster_comparison import paired_cluster_interval


def test_paired_cluster_interval_preserves_pairing_and_reproducibility():
    a = [1, 2, 3, 4, 5, 6]
    b = [2, 3, 4, 5, 6, 7]
    subjects = ["a", "a", "b", "b", "c", "c"]
    result = paired_cluster_interval(a, b, subjects)
    assert result == paired_cluster_interval(a, b, subjects)
    assert result["mean_difference"] == -1
    assert result["interval_95"] == [-1, -1]
    assert result["clusters"] == 3


def test_cluster_uncertainty_does_not_shrink_by_duplicating_trials():
    a = np.array([0., 2., 4.])
    b = np.zeros(3)
    groups = np.array(["a", "b", "c"])
    original = paired_cluster_interval(a, b, groups)
    repeated = paired_cluster_interval(np.repeat(a, 10), np.repeat(b, 10), np.repeat(groups, 10))
    np.testing.assert_allclose(original["interval_95"], repeated["interval_95"])


@pytest.mark.parametrize("a,b,g", [([], [], []), ([1], [2], ["a"]),
                                 ([1, 2], [2], ["a", "b"]),
                                 ([1, float("nan")], [2, 3], ["a", "b"])])
def test_bad_pairs_fail(a, b, g):
    with pytest.raises(ValueError):
        paired_cluster_interval(a, b, g)
