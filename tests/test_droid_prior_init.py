import numpy as np

from scripts.probe_droid_prior_init import fit_predict


def test_zero_update_matches_training_prior():
    labels = np.array([0, 0, 1, 2, 3, 4, 5, 6, 6, 6])
    fit = np.arange(10) < 8
    data = np.arange(20, dtype=np.float32).reshape(10, 2)
    output = fit_predict(data, data, data, labels, fit, ~fit, 'sensor', (0,))[0]
    expected = np.bincount(labels[fit], minlength=7) / fit.sum()
    assert np.allclose(output, np.tile(expected, (2, 1)))
