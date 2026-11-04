import numpy as np
import pytest

from scripts.egoviz_expanded_oof_calibration import probabilities, training_folds


def test_expanded_training_folds_are_deterministic_and_group_disjoint():
    groups = np.repeat([f"g{i:02d}" for i in range(48)], [16] * 47 + [21])
    first = training_folds(groups)
    second = training_folds(groups)
    assert all(
        np.array_equal(a, b)
        for fold_a, fold_b in zip(first, second, strict=True)
        for a, b in zip(fold_a, fold_b, strict=True)
    )
    heldout_coverage = np.zeros(len(groups), dtype=np.int64)
    for fit, inner, held in first:
        assert not np.any(fit & inner)
        assert not np.any(fit & held)
        assert not np.any(inner & held)
        assert np.all(fit | inner | held)
        heldout_coverage += held
    assert np.all(heldout_coverage == 1)


@pytest.mark.parametrize(
    "groups", [np.array([]), np.array(["a", "b", "c", "d"]), np.array(["", "a", "b", "c", "d"])]
)
def test_expanded_training_folds_reject_invalid_group_ids(groups):
    with pytest.raises(ValueError):
        training_folds(groups)


def test_positive_temperature_changes_confidence_but_preserves_rankings():
    logits = np.array([[0.2, 0.5, -0.4, 0.1], [0.8, 0.3, 0.2, 0.1]])
    alpha = 3.5
    base = probabilities(logits)
    calibrated = probabilities(alpha * logits)
    assert np.array_equal(base.argmax(axis=1), calibrated.argmax(axis=1))
    assert not np.allclose(base, calibrated)
    assert np.allclose(calibrated.sum(axis=1), 1.0)
