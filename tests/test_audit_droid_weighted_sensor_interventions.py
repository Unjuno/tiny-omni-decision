import numpy as np
import pytest

from scripts.audit_droid_weighted_sensor_interventions import (
    group_bootstrap,
    within_recording_shift,
)


def test_shift_is_disjoint_within_each_recording():
    groups = [{"start": 0, "end": 24}, {"start": 24, "end": 48}]
    donors = within_recording_shift(groups, 48)
    assert sorted(donors.tolist()) == list(range(48))
    assert all(abs(i - donor) >= 8 for i, donor in enumerate(donors))
    assert np.all(donors[:24] < 24) and np.all(donors[24:] >= 24)
    with pytest.raises(ValueError, match="invalid"):
        within_recording_shift([{"start": 0, "end": 23}], 23)


def test_bootstrap_is_paired_and_deterministic():
    group = np.arange(275)
    targets = np.zeros(275, dtype=int)
    p = np.tile(np.array([0.4, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]), (275, 1))
    result = group_bootstrap(targets, p, p, group, replicates=20)
    assert result == group_bootstrap(targets, p, p, group, replicates=20)
    assert result["nll_95_percentile_interval"] == [0.0, 0.0]
    assert result["accuracy_95_percentile_interval"] == [0.0, 0.0]
