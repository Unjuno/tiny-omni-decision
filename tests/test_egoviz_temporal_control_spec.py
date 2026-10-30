import numpy as np
import pytest

from scripts.diagnose_egoviz_temporal_controls import CONTROLS, control_spec


def test_controls_are_fixed_deterministic_and_keep_all_options() -> None:
    assert len(CONTROLS) == 7
    for name in CONTROLS:
        spec = control_spec(name)
        assert spec == control_spec(name)
        assert sorted(spec["video_frame_indices"]) == list(range(8))
        assert sorted(spec["sensor_patch_indices"]) == list(range(16))
        assert "option_indices" not in spec


def test_reversing_both_retains_adjacent_sensor_patch_pair_alignment() -> None:
    spec = control_spec("reverse_both_features")
    sensor_slots = np.asarray(spec["sensor_patch_indices"]).reshape(8, 2) // 2
    expected = np.asarray(spec["video_frame_indices"])
    np.testing.assert_array_equal(sensor_slots[:, 0], expected)
    np.testing.assert_array_equal(sensor_slots[:, 1], expected)


def test_sensor_shift_is_one_video_frame_and_preserves_patch_pairs() -> None:
    spec = control_spec("shift_sensor_features_one_frame")
    assert spec["video_frame_indices"] == list(range(8))
    assert spec["sensor_patch_indices"] == [14, 15, *range(14)]


def test_missing_input_flags_are_separate_from_order_changes() -> None:
    for name in CONTROLS:
        spec = control_spec(name)
        assert spec["zero_video"] == (name == "missing_video_features")
        assert spec["zero_sensor"] == (name == "missing_sensor_features")


def test_unknown_control_fails_closed() -> None:
    with pytest.raises(ValueError, match="unknown temporal control"):
        control_spec("random_search_for_a_better_score")
