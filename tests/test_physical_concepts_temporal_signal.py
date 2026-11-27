from __future__ import annotations

import pytest

from scripts.audit_physical_concepts_temporal_signal import sequence_signal

np = pytest.importorskip("numpy")


def test_sequence_signal_detects_rgb_mask_and_pose_change() -> None:
    rgb = np.zeros((15, 64, 64, 3), dtype=np.uint8)
    rgb[1:] = 255
    mask = np.zeros((15, 64, 64), dtype=np.uint8)
    mask[1:] = 1
    pose = np.zeros((15, 6), dtype=np.float32)
    pose[:, 0] = np.arange(15, dtype=np.float32)

    result = sequence_signal(rgb.tobytes(), mask.tobytes(), pose.reshape(-1).tolist(), np)

    assert result["rgb_adjacent_mae_mean"] == pytest.approx(1 / 14)
    assert result["rgb_identical_transition_count"] == 13
    assert result["unique_rgb_frames"] == 2
    assert result["mask_changed_transition_count"] == 1
    assert result["unique_mask_frames"] == 2
    assert result["camera_pose_delta_l2_mean"] == pytest.approx(1.0)
    assert result["unique_camera_pose_rows"] == 15


def test_sequence_signal_reports_static_sequence() -> None:
    rgb = np.zeros((15, 64, 64, 3), dtype=np.uint8)
    mask = np.zeros((15, 64, 64), dtype=np.uint8)
    pose = np.zeros((15, 6), dtype=np.float32)

    result = sequence_signal(rgb.tobytes(), mask.tobytes(), pose.reshape(-1).tolist(), np)

    assert result["rgb_adjacent_mae_mean"] == 0
    assert result["mask_changed_transition_count"] == 0
    assert result["camera_pose_static_transition_count"] == 14


def test_sequence_signal_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="unexpected Physical Concepts image/mask shape"):
        sequence_signal(b"bad", b"bad", [0.0] * 90, np)
