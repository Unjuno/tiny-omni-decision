import numpy as np
import pytest

from scripts.egoviz_train_oof_folds import training_folds


def test_each_session_is_held_out_once_and_excluded_from_inner_and_fit():
    groups = np.array([f"g{i:02d}" for i in range(24) for _ in range(3)])
    folds = training_folds(groups)
    assert len(folds) == 5
    assert np.all(sum(held for _, _, held in folds) == 1)
    for fit, inner, held in folds:
        assert np.all(fit.astype(int) + inner + held == 1)
        assert not (set(groups[fit]) & set(groups[held]))
        assert not (set(groups[inner]) & set(groups[held]))
    for old, new in zip(folds, training_folds(groups), strict=True):
        for x, y in zip(old, new, strict=True):
            np.testing.assert_array_equal(x, y)


def test_wrong_group_count_fails_closed():
    with pytest.raises(ValueError, match="24 training"):
        training_folds(np.array(["one", "two"]))
