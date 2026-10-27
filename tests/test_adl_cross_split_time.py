import numpy as np
import pytest

from scripts.audit_adl_cross_split_time import nearest_cross_split_gaps


def test_returns_nearest_gap_for_every_validation_event_in_original_order():
    np.testing.assert_array_equal(
        nearest_cross_split_gaps([3.0, 1.0], [3.0, 2.0, 9.0]), [0.0, 1.0, 6.0]
    )


def test_empty_and_nonfinite_input_fail_closed():
    with pytest.raises(ValueError):
        nearest_cross_split_gaps([], [0.0])
    with pytest.raises(ValueError):
        nearest_cross_split_gaps([0.0], [float('nan')])
