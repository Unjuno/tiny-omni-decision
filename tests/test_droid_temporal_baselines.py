import numpy as np
import pytest

from scripts.audit_droid_train_temporal import direction, metrics, temporal_predictions


@pytest.mark.parametrize(
    ('vector', 'expected'),
    [([0, 0, 0], 0), ([1e-4, 0, 0], 0), ([-1, 1, 0], 1),
     ([1, 1, 1], 2), ([0, -2, 0], 3), ([0, 2, 0], 4),
     ([0, 0, -3], 5), ([0, 0, 3], 6)],
)
def test_direction_matches_frozen_stop_sign_and_first_axis_ties(vector, expected):
    assert direction(vector) == expected


def test_baselines_use_only_past_and_distinguish_sparse_from_dense():
    positions = np.zeros((10, 6))
    positions[3, 0] = 2
    positions[4, 1] = 3
    actions = np.zeros((10, 6))
    actions[4, 2] = -1
    expected = {'previous_command': 5, 'sparse_displacement': 4, 'dense_displacement': 4}
    assert temporal_predictions(positions, actions, [0, 2, 4], 5) == expected
    positions[5:] = 987
    actions[5:] = -987
    assert temporal_predictions(positions, actions, [0, 2, 4], 5) == expected
    positions[3, 0] = 5
    assert temporal_predictions(positions, actions, [0, 2, 4], 5) == {
        **expected, 'dense_displacement': 1,
    }


@pytest.mark.parametrize('indices,target', [([0, 4, 5], 5), ([0, 4, 3], 5), ([0], 1)])
def test_rejects_future_unsorted_or_insufficient_indices(indices, target):
    with pytest.raises(ValueError):
        temporal_predictions(np.zeros((8, 6)), np.zeros((8, 6)), indices, target)


def test_rejects_nonfinite_direction():
    with pytest.raises(ValueError):
        direction([np.nan, 0, 0])


def test_metrics_count_majority_without_hiding_minority_failure():
    result = metrics([0, 0, 2], [0, 0, 0])
    assert result['accuracy'] == 2 / 3
    assert result['balanced_accuracy_present_classes'] == 0.5
    assert result['per_class']['x+'] == {'count': 1, 'accuracy': 0.0}
    assert result['confusion_matrix_target_rows'][2][0] == 1
