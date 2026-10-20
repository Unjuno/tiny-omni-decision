from __future__ import annotations

import numpy as np
import pytest

from scripts.audit_holoassist_sensor_coverage import audit_split


def _rows() -> list[dict]:
    return [
        {
            "id": "session-a:window-0000",
            "window_start_seconds": 0.0,
            "window_end_seconds": 30.0,
            "window_index": 0,
            "window_count": 2,
            "target": 1,
            "action_count": 4,
        },
        {
            "id": "session-a:window-0001",
            "window_start_seconds": 30.0,
            "window_end_seconds": 35.0,
            "window_index": 1,
            "window_count": 2,
            "target": 0,
            "action_count": 1,
        },
    ]


def test_audit_separates_zero_padding_from_zero_observed_coverage() -> None:
    features = np.zeros((2, 30, 21), dtype=np.float32)
    features[0, :, 18:21] = 1.0
    features[1, :5, 18:21] = 1.0
    # An actually unobserved bin among the valid observations.
    features[0, 7, 18:21] = 0.0

    result = audit_split(
        _rows(), np.asarray([row["id"] for row in _rows()]), features
    )

    assert result["coverage_bins"]["zero_padding"]["zero_coverage_bin_count"] == 25
    assert result["coverage_bins"]["valid_observation"]["zero_coverage_bin_count"] == 1
    assert result["coverage_bins"]["zero_bins_explained_by_padding_percent"] == pytest.approx(
        25 / 26 * 100
    )
    assert result["labeled_window_groups"]["final_window"]["mean_padded_bins"] == 25


def test_audit_rejects_misaligned_sample_ids() -> None:
    features = np.zeros((2, 30, 21), dtype=np.float32)
    with pytest.raises(ValueError, match="sample ID order differ"):
        audit_split(_rows(), np.asarray(["wrong-1", "wrong-2"]), features)


def test_audit_rejects_coverage_channel_mismatch() -> None:
    features = np.zeros((2, 30, 21), dtype=np.float32)
    features[0, 0, 18] = 1.0
    with pytest.raises(ValueError, match="coverage channels unexpectedly differ"):
        audit_split(_rows(), np.asarray([row["id"] for row in _rows()]), features)
