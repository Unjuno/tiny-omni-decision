import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.extract_eidon_sensor_features import VIDEO_REVISION, _load_split, midpoint_window


def test_midpoint_window_accounts_for_first_sensor_timestamp():
    assert midpoint_window(15.0, 125.0) == (5125.0, 10125.0)


@pytest.mark.parametrize("duration,first,window", [(0, 0, 5), (4, 0, 5), (10, -1, 5),
                                                    (10, 0, 0), (10, 0, 11)])
def test_midpoint_window_rejects_invalid_or_uncontained_windows(duration, first, window):
    with pytest.raises(ValueError):
        midpoint_window(duration, first, window)


def test_sensor_extraction_loads_only_pinned_train_development_manifest(tmp_path):
    path = tmp_path / "train-development-manifest-private.json"
    path.write_text(json.dumps({
        "video_revision": VIDEO_REVISION,
        "train": [{"recording_id": 1, "has_raw_motion": True}],
        "development": [{"recording_id": 2, "has_raw_motion": True}],
    }))
    metadata = tmp_path / "metadata.parquet"
    pq.write_table(pa.table({
        "recording_id": [1, 2], "qc_status": ["valid", "valid"],
        "n_slots": [7, 7], "has_chest": [True, True],
        "duration_seconds": [10.0, 10.0],
        "video_duration_seconds": [9.5, 9.5],
    }), metadata)
    assert len(_load_split(path, metadata, "train")) == 1
    assert len(_load_split(path, metadata, "development")) == 1
    with pytest.raises(ValueError, match="only train and development"):
        _load_split(path, metadata, "sealed_audit")
    audit_path = tmp_path / "sealed-audit-manifest-private.json"
    audit_path.write_text(path.read_text())
    with pytest.raises(ValueError, match="only the frozen train/development"):
        _load_split(audit_path, metadata, "development")


def test_no_raw_accel_or_gyro_flag_does_not_exclude_quaternion_rig(tmp_path):
    path = tmp_path / "train-development-manifest-private.json"
    path.write_text(json.dumps({"video_revision": VIDEO_REVISION,
                                "train": [{"recording_id": 1, "has_raw_motion": False}],
                                "development": [{"recording_id": 1, "has_raw_motion": False}]}))
    metadata = tmp_path / "metadata.parquet"
    pq.write_table(pa.table({
        "recording_id": [1], "qc_status": ["valid"], "n_slots": [7],
        "has_chest": [True], "duration_seconds": [10.0],
        "video_duration_seconds": [9.5],
    }), metadata)
    assert _load_split(path, metadata, "train")[0]["imu_duration_seconds"] == 10.0
