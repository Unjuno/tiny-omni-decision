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
    assert result["overlap_with_droid100_debug"] == {
        "source_files": 0,
        "recording_groups": 1,
        "normalized_instructions": 1,
    }
    assert result["success_failure_source_folders"] == {"success": 1, "failure": 1}
