import numpy as np
import pytest

from scripts.audit_touch_and_go_metadata import audit_labels, timestamp_summary


def test_frame_count_is_not_recording_count_and_rare_class_stays_visible():
    rows = [
        "20220318_120000/0000000001.jpg,3",
        "20220318_120000/0000000002.jpg,3",
        "20220318_120000/0000000003.jpg,-1",
        "20220319_120000/0000000001.jpg,9",
    ]
    result = audit_labels(rows, {-1: "Inconclusive", 3: "Wood", 9: "Natural Fabric"})
    assert result["frame_rows"] == 4
    assert result["recording_count"] == 2
    assert result["date_prefix_count"] == 2
    assert result["inconclusive_frames"] == 1
    assert result["classes"]["9"]["frames"] == 1
    assert result["classes"]["9"]["recordings"] == 1
    assert not result["all_material_classes_can_cover_four_recording_partitions"]
    assert not result["independent_sessions_verified"]


@pytest.mark.parametrize(
    "rows",
    [
        ["20220318_120000/0000000001.jpg,3"] * 2,
        ["20220318_120000/0000000001.jpg,3", "20220318_120000/0000000001.jpg,9"],
        ["../../escape.jpg,3"],
        ["20229999_120000/0000000001.jpg,3"],
        ["20220318_120000/0000000001.jpg,100"],
    ],
)
def test_bad_or_duplicate_annotation_identity_fails_closed(rows):
    with pytest.raises(ValueError):
        audit_labels(rows, {-1: "Inconclusive", 3: "Wood", 9: "Natural Fabric"})


def test_timestamp_agreement_does_not_certify_capture_sync():
    result = timestamp_summary(np.array([1.0, 2.0, 3.0]), np.array([1.05, 2.1, 3.05]))
    assert result["index_offset_abs_max_native_units"] == pytest.approx(0.1)
    assert not result["physical_synchronization_verified"]
    assert result["timestamp_units_verified"] is False
    with pytest.raises(ValueError):
        timestamp_summary(np.array([1.0, 1.0]), np.array([1.0, 2.0]))
    with pytest.raises(ValueError):
        timestamp_summary(np.array([1.0, np.nan]), np.array([1.0, 2.0]))
