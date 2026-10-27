import numpy as np

from scripts.audit_droid_prior_uncertainty import bootstrap, paired_deltas


def test_identical_probabilities_have_zero_difference_and_interval():
    labels = np.array([0, 1, 0])
    p = np.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4]])
    delta = paired_deltas(labels, p, p)
    assert np.allclose(delta, 0)
    assert np.allclose(bootstrap(delta, np.array([1, 1, 2]), repeats=20), 0)
