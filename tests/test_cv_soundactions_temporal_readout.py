import numpy as np
import pytest

from scripts.cv_soundactions_temporal_readout import stratified_folds


def test_folds_are_deterministic_complete_disjoint_and_class_stratified():
    targets = np.repeat(np.arange(3), [41, 53, 67])
    first = stratified_folds(targets, fold_count=5, seed=17)
    second = stratified_folds(targets, fold_count=5, seed=17)
    assert np.array_equal(first, second)
    assert np.bincount(first, minlength=5).sum() == len(targets)
    assert set(first.tolist()) == set(range(5))
    for fold in range(5):
        assert set(targets[first == fold].tolist()) == {0, 1, 2}


def test_folds_reject_missing_class_in_small_fold():
    with pytest.raises(ValueError, match="every target class"):
        stratified_folds(np.array([0, 0, 1, 1, 2]), fold_count=4)


def test_rows_from_one_recording_stay_in_one_fold():
    groups = np.repeat([f"group-{i}" for i in range(9)], 2)
    targets = np.repeat(np.repeat(np.arange(3), 3), 2)
    folds = stratified_folds(targets, group_ids=groups, fold_count=3, seed=17)
    for group in np.unique(groups):
        assert len(np.unique(folds[groups == group])) == 1


def test_group_with_conflicting_targets_is_rejected():
    with pytest.raises(ValueError, match="same target"):
        stratified_folds(
            np.array([0, 1, 0, 1, 2, 2]),
            group_ids=np.array(["a", "a", "b", "c", "d", "e"]),
            fold_count=3,
        )
