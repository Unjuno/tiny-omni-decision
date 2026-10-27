import numpy as np
import pytest

from scripts.build_droid_dense_train_windows import fixed_folds, make_windows


def test_windows_exclude_target_and_future_inputs():
    sensor = np.arange(12 * 14, dtype=np.float32).reshape(12, 14)
    actions = np.zeros((12, 6), dtype=np.float64)
    actions[7, 0] = 1
    actions[8, 1] = -1
    windows, previous, targets = make_windows(sensor, actions)
    assert windows.shape == (4, 8, 14)
    assert np.array_equal(windows[0], sensor[:8])
    assert previous[0] == 2 and targets[0] == 3
    changed_sensor, changed_actions = sensor.copy(), actions.copy()
    changed_sensor[8:] = -42
    changed_actions[8:] = 17
    modified, previous_modified, _ = make_windows(changed_sensor, changed_actions)
    assert np.array_equal(modified[0], windows[0])
    assert previous_modified[0] == previous[0]


def test_rejects_invalid_fold_or_nonfinite_source():
    with pytest.raises(ValueError, match="finite"):
        make_windows(np.full((10, 14), np.nan), np.zeros((10, 6)))
    rows = [{"id": str(i), "group_id": str(i)} for i in range(5)]
    folds = [{"held_indices": [i], "train_indices": [j for j in range(5) if j != i]}
             for i in range(5)]
    assert list(fixed_folds(rows, rows, folds)) == list(range(5))
    folds[0]["train_indices"].append(0)
    with pytest.raises(ValueError, match="partition"):
        fixed_folds(rows, rows, folds)
