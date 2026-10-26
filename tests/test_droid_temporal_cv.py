import numpy as np
import pytest

from scripts.compare_droid_train_temporal import nested_compare, temporal_features


def test_transform_is_invertible_and_has_equal_width():
    x = np.arange(2 * 8 * 14, dtype=float).reshape(2, 8, 14)
    arms = temporal_features(x)
    assert arms['raw'].shape == arms['delta'].shape == (2, 112)
    delta = arms['delta'][:, 14:].reshape(2, 7, 14)
    last = arms['delta'][:, :14]
    restored = np.concatenate([
        last[:, None] - np.cumsum(delta[:, ::-1], axis=1)[:, ::-1],
        last[:, None],
    ], axis=1)
    np.testing.assert_array_equal(restored, x)
    np.testing.assert_array_equal(arms['delta'][:, :14], x[:, -1])


@pytest.mark.parametrize('x', [np.zeros((2, 7, 14)), np.full((2, 8, 14), np.nan)])
def test_rejects_invalid_sensor_sequences(x):
    with pytest.raises(ValueError):
        temporal_features(x)


def test_outer_test_rows_cannot_enter_selection_or_fit():
    x = np.arange(12, dtype=float).reshape(6, 2)
    y = np.array([0, 1, 0, 1, 0, 1])
    groups = [str(i) for i in range(6)]
    selections, fits = [], []

    def folds(targets, group_ids, **kwargs):
        return [np.array([0, 1]), np.array([2, 3]), np.array([4, 5])]

    def select(features, targets, group_ids, **kwargs):
        selections.append((features.copy(), list(group_ids)))
        return {'selected_l2_strength': 100.0}

    def fit(features, targets, **kwargs):
        fits.append(features.copy())
        return {'training_rows': features[:, 0].tolist()}

    def predict(model, features):
        assert not set(features[:, 0]) & set(model['training_rows'])
        return np.full((len(features), 7), 1 / 7)

    api = {'stratified_group_folds': folds, 'select_l2_strength_group_cv': select,
           'fit_l2_softmax': fit, 'predict_softmax': predict}
    result = nested_compare({'raw': x, 'delta': x + 100}, y, groups, api)
    assert len(selections) == len(fits) == 6
    assert result['probabilities']['raw'].shape == (6, 7)
    for selection, fitted in zip(selections, fits, strict=True):
        np.testing.assert_array_equal(selection[0], fitted)
        assert len(selection[1]) == 4
