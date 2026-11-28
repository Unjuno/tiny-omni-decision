from __future__ import annotations

import numpy as np
import pytest

from scripts.audit_physionpp_low_level_pair_cues import (
    frame_pixel_features,
    paired_permutation_summary,
    uniform_indices,
)


def test_uniform_indices_are_deterministic_and_endpoint_inclusive() -> None:
    assert uniform_indices(29) == [0, 4, 8, 12, 16, 20, 24, 28]
    assert uniform_indices(10, 3) == [0, 4, 9]


def test_uniform_indices_reject_invalid_counts() -> None:
    with pytest.raises(ValueError):
        uniform_indices(1)
    with pytest.raises(ValueError):
        uniform_indices(12, 1)


def test_frame_pixel_features_detect_fixed_rgb_values() -> None:
    red = np.zeros((8, 4, 4, 3), dtype=np.uint8)
    red[..., 0] = 255
    result = frame_pixel_features([frame for frame in red])
    assert result["mean_red"] == 1.0
    assert result["mean_green"] == 0.0
    assert result["mean_blue"] == 0.0
    assert result["mean_adjacent_sample_luma_delta"] == 0.0
    assert result["clipped_pixel_fraction"] == 1.0


def test_frame_pixel_features_reject_invalid_shape_or_too_few_frames() -> None:
    with pytest.raises(ValueError):
        frame_pixel_features([np.zeros((2, 2, 3), dtype=np.uint8)])
    with pytest.raises(ValueError):
        frame_pixel_features([np.zeros((2, 2), dtype=np.uint8)] * 2)


def test_paired_permutation_summary_is_seeded_and_reports_uncertainty() -> None:
    first = paired_permutation_summary([0.1] * 12)
    second = paired_permutation_summary([0.1] * 12)
    assert first == second
    assert first["n_pairs"] == 12
    assert first["mean_yes_minus_no"] == pytest.approx(0.1)
    assert first["bootstrap_95pct_ci_mean"] == pytest.approx([0.1, 0.1])
    assert first["paired_sign_flip_p_value"] < 0.01


def test_paired_permutation_summary_rejects_empty_or_nonfinite_input() -> None:
    with pytest.raises(ValueError):
        paired_permutation_summary([])
    with pytest.raises(ValueError):
        paired_permutation_summary([0.1, float("nan")])
