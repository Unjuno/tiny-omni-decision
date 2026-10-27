import numpy as np
import pytest

from scripts.audit_droid_train_all_transitions import transition_counts


def test_counts_changes_after_minimum_history():
    actions = np.zeros((11, 6))
    actions[8:, 0] = 1.0
    actions[10, 0] = -1.0
    matrix = transition_counts(actions)
    assert matrix.sum() == 3
    assert matrix[0, 2] == 1
    assert matrix[2, 2] == 1
    assert matrix[2, 1] == 1


def test_rejects_nonfinite_and_missing_history():
    with pytest.raises(ValueError, match="finite"):
        transition_counts(np.full((9, 6), np.nan))
    with pytest.raises(ValueError, match="follow"):
        transition_counts(np.zeros((9, 6)), minimum_target=0)
