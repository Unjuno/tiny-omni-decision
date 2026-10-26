import pytest

from scripts.audit_droid_selected_shards import audit_selected_shards, normalize_instruction


def _varint(value: int) -> bytes:
    result = bytearray()
    while value > 127:
        result.append((value & 127) | 128)
        value >>= 7
    result.append(value)
    return bytes(result)


def _field(number: int, value: bytes) -> bytes:
    return _varint((number << 3) | 2) + _varint(len(value)) + value


def _episode(path: str, group: str, instruction: str) -> bytes:
    entries = []
    for key, value in (
        ("episode_metadata/file_path", path),
        ("episode_metadata/recording_folderpath", group),
        ("steps/language_instruction", instruction),
    ):
        bytes_list = _field(1, value.encode())
        feature = _field(1, bytes_list)
        entries.append(_field(1, _field(1, key.encode()) + _field(2, feature)))
    return _field(1, b"".join(entries))


def test_instruction_normalization_ignores_case_punctuation_and_spacing():
    assert normalize_instruction(" Put the cup,   left! ") == "put the cup left"


def test_audit_counts_groups_and_debug_overlap(monkeypatch, tmp_path):
    records = [
        _episode("/source/success/a", "/recording/one", "Put the cup, left!"),
        _episode("/source/failure/b", "/recording/one", "put the cup left"),
    ]
    monkeypatch.setattr("scripts.audit_droid_selected_shards.validate_manifest", lambda _: None)
    monkeypatch.setattr("scripts.audit_droid_selected_shards.verify_metadata", lambda *_: None)
    monkeypatch.setattr(
        "scripts.audit_droid_selected_shards.episode_example_feature_counts",
        lambda _: {
            "steps/is_first": {"count": 1},
            "steps/observation/exterior_image_1_left": {"count": 1},
            "steps/observation/exterior_image_2_left": {"count": 1},
            "steps/observation/wrist_image_left": {"count": 1},
            "steps/action_dict/joint_velocity": {"count": 7},
            "steps/action_dict/gripper_position": {"count": 1},
            "steps/language_instruction": {"count": 1, "nonempty": 1},
        },
    )
    monkeypatch.setattr(
        "scripts.audit_droid_selected_shards.iter_tfrecord",
        lambda _: iter(records),
    )
    manifest = {
        "dataset_id": "droid/droid",
        "dataset_version": "1.4.0",
        "selection_seed": 17,
        "expected_episode_count": 2,
        "selected_shards": [{
            "index": 110,
            "episodes": 2,
            "object_name": "robotics/droid/1.0.0/r2d2_faceblur-train.tfrecord-00110-of-02048",
        }],
    }
    debug = {
        "episodes": [{
            "episode_metadata/file_path": "/source/old",
            "episode_metadata/recording_folderpath": "/recording/one",
        }],
        "instruction_episode_frequency": {"PUT the cup left.": 1},
    }
    result = audit_selected_shards(tmp_path, manifest, debug)
    assert result["unique_source_files"] == 2
    assert result["unique_recording_groups"] == 1
    assert result["duplicate_recording_groups"] == 1
    assert result["normalized_instruction_repeats"] == 1
    assert result["instruction_presence"] == {
        "episodes_with_instruction": 2,
        "episodes_without_instruction": 0,
    }
    assert result["step_count_distribution"] == {
        "min": 1,
        "median": 1,
        "p90": 1,
        "max": 1,
        "mean": 1.0,
    }
    assert result["overlap_with_droid100_debug"] == {
        "source_files": 0,
        "recording_groups": 1,
        "normalized_instructions": 1,
    }
    assert result["success_failure_source_folders"] == {"success": 1, "failure": 1}


def test_audit_rejects_camera_frame_count_mismatch(monkeypatch, tmp_path):
    record = _episode("/source/success/a", "/recording/one", "Move")
    monkeypatch.setattr("scripts.audit_droid_selected_shards.validate_manifest", lambda _: None)
    monkeypatch.setattr("scripts.audit_droid_selected_shards.verify_metadata", lambda *_: None)
    monkeypatch.setattr(
        "scripts.audit_droid_selected_shards.episode_example_feature_counts",
        lambda _: {
            "steps/is_first": {"count": 2},
            "steps/observation/exterior_image_1_left": {"count": 1},
            "steps/observation/exterior_image_2_left": {"count": 2},
            "steps/observation/wrist_image_left": {"count": 2},
            "steps/action_dict/joint_velocity": {"count": 14},
            "steps/action_dict/gripper_position": {"count": 2},
            "steps/language_instruction": {"count": 2, "nonempty": 2},
        },
    )
    monkeypatch.setattr(
        "scripts.audit_droid_selected_shards.iter_tfrecord", lambda _: iter([record])
    )
    manifest = {
        "dataset_id": "droid/droid",
        "dataset_version": "1.4.0",
        "selection_seed": 17,
        "expected_episode_count": 1,
        "selected_shards": [{
            "index": 110,
            "episodes": 1,
            "object_name": "robotics/droid/1.0.0/sample.tfrecord",
        }],
    }

    with pytest.raises(ValueError, match="camera frame count"):
        audit_selected_shards(tmp_path, manifest, {"episodes": []})
