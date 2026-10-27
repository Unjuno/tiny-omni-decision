import numpy as np
import pytest

from scripts.audit_adl_sensor_identity import compare_sensor_rows


def test_aligns_by_participant_and_trial_not_row_order():
    ref = (np.array([1, 1, 2]), np.array([10, 11, 1]), np.array([2., 3., 4.]))
    other = (np.array([2, 1, 1]), np.array([1, 10, 11]), np.array([4.2, 2.1, 3.3]))
    result = compare_sensor_rows(ref, other)
    assert result['common_pairs'] == 3
    assert result['reference_only_pairs'] == result['sensor_only_pairs'] == 0
    assert result['max_abs_timestamp_delta_native_units'] == pytest.approx(0.3)


def test_fails_on_duplicate_identity_or_nonfinite_timestamp():
    ref = ([1, 1], [0, 1], [2., 3.])
    with pytest.raises(ValueError):
        compare_sensor_rows(ref, ([1, 1], [0, 0], [1., 2.]))
    with pytest.raises(ValueError):
        compare_sensor_rows(ref, ([1, 1], [0, 1], [float('nan'), 2.]))


def test_reports_unmatched_recording_pairs():
    result = compare_sensor_rows(([1], [1], [0.]), ([2], [1], [0.]))
    assert result['common_pairs'] == 0
    assert result['reference_only_pairs'] == result['sensor_only_pairs'] == 1
    assert result['max_abs_timestamp_delta_native_units'] is None
