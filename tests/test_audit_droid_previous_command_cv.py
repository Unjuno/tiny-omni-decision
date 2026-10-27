import numpy as np
import pytest

pytest.importorskip("torch")
from scripts.audit_droid_previous_command_cv import transition_oof  # noqa: E402


def test_held_targets_do_not_change_own_fold_probabilities():
    previous = [1, 1, 2, 2]
    targets = [1, 2, 2, 3]
    folds = [{"train_indices": [2, 3], "held_indices": [0, 1]},
             {"train_indices": [0, 1], "held_indices": [2, 3]}]
    first = transition_oof(previous, targets, folds)
    changed = transition_oof(previous, [4, 5, 2, 3], folds)
    assert np.allclose(first[:2], changed[:2])
    assert np.allclose(first.sum(axis=1), 1)
    assert np.all(first > 0)


def test_rejects_fold_leakage_and_out_of_range():
    with pytest.raises(ValueError, match="partition"):
        transition_oof([0, 1], [0, 1], [{"train_indices": [0], "held_indices": [0, 1]}])
    with pytest.raises(ValueError, match="seven-class"):
        transition_oof([7], [0], [])
