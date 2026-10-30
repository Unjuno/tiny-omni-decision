import numpy as np

from scripts.egoviz_temperature_math import cross_fit, fit_alpha


def test_temperature_fits_known_optimum_and_boundaries():
    logits = np.tile([1.0, 0.0, -100.0, -100.0], (4, 1))
    assert abs(fit_alpha(logits, np.array([0, 0, 0, 1])) - np.log(3)) < 1e-8
    assert fit_alpha(logits, np.zeros(4, dtype=int)) == 20.0
    boundary_logits = np.tile([1.0, 0.0, 0.0, 0.0], (4, 1))
    assert fit_alpha(boundary_logits, np.ones(4, dtype=int)) == 0.05


def test_heldout_group_labels_do_not_affect_its_fold_temperature():
    groups = np.array([f"g{i:02d}" for i in range(20) for _ in range(2)])
    logits = np.tile([0.7, 0.3, -0.3, -0.5], (len(groups), 1))
    targets = np.arange(len(groups)) % 4
    output, folds = cross_fit(logits, targets, groups)
    changed = targets.copy()
    changed[np.isin(groups, folds[0]["heldout_groups"])] = 0
    changed_output, changed_folds = cross_fit(logits, changed, groups)
    mask = np.isin(groups, folds[0]["heldout_groups"])
    assert folds[0]["alpha"] == changed_folds[0]["alpha"]
    np.testing.assert_array_equal(output[mask], changed_output[mask])
    np.testing.assert_array_equal(output.argmax(axis=1), logits.argmax(axis=1))
    assert len({g for fold in folds for g in fold["heldout_groups"]}) == 20
