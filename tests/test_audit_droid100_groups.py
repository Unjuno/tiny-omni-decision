import struct

import pytest

from scripts.audit_droid100_groups import (
    audit_dataset,
    episode_example_context,
    episode_example_feature_counts,
    episode_example_instructions,
    iter_tfrecord,
)


def _varint(value: int) -> bytes:
    encoded = bytearray()
    while value > 0x7F:
        encoded.append((value & 0x7F) | 0x80)
        value >>= 7
    encoded.append(value)
    return bytes(encoded)


def _length_field(number: int, value: bytes) -> bytes:
    return _varint((number << 3) | 2) + _varint(len(value)) + value


def _episode_example(file_path: str, recording_path: str, instruction: str | None = None) -> bytes:
    entries = []
    for key, value in (
        ("episode_metadata/file_path", file_path),
        ("episode_metadata/recording_folderpath", recording_path),
    ):
        bytes_list = _length_field(1, value.encode())
        feature = _length_field(1, bytes_list)
        entry = _length_field(1, key.encode()) + _length_field(2, feature)
        entries.append(_length_field(1, entry))
    if instruction is not None:
        bytes_list = _length_field(1, instruction.encode())
        feature = _length_field(1, bytes_list)
        entry = _length_field(1, b"steps/language_instruction") + _length_field(2, feature)
        entries.append(_length_field(1, entry))
    return _length_field(1, b"".join(entries))


def _write_tfrecord(path, payload: bytes):
    path.write_bytes(struct.pack("<Q", len(payload)) + bytes(4) + payload + bytes(4))


def test_episode_example_reads_only_recording_identity_metadata():
    row = episode_example_context(_episode_example("episode/1", "recording/abc"))
    assert row == {
        "episode_metadata/file_path": "episode/1",
        "episode_metadata/recording_folderpath": "recording/abc",
    }


def test_episode_example_rejects_missing_group_identity():
    with pytest.raises(ValueError, match="missing non-empty"):
        episode_example_context(_length_field(1, b""))


def test_episode_example_extracts_normalized_instruction_text():
    row = _episode_example("episode/1", "recording/abc", "  Move   the   cup\n left ")
    assert episode_example_instructions(row) == ["Move the cup left"]


def test_feature_count_audit_handles_empty_bytes_and_packed_float_steps():
    bytes_list = _length_field(1, b"") + _length_field(1, b"label")
    bytes_feature = _length_field(1, bytes_list)
    float_list = _length_field(1, struct.pack("<ff", 0.25, -0.5))
    float_feature = _length_field(2, float_list)
    entries = []
    for key, feature in (
        ("steps/language_instruction", bytes_feature),
        ("steps/action_dict/joint_velocity", float_feature),
    ):
        entry = _length_field(1, key.encode()) + _length_field(2, feature)
        entries.append(_length_field(1, entry))
    example = _length_field(1, b"".join(entries))
    assert episode_example_feature_counts(example) == {
        "steps/language_instruction": {"kind": "bytes", "count": 2, "nonempty": 1},
        "steps/action_dict/joint_velocity": {"kind": "float", "count": 2},
    }


def test_tfrecord_reader_checks_record_framing(tmp_path):
    payload = _episode_example("episode/1", "recording/abc")
    path = tmp_path / "sample.tfrecord"
    _write_tfrecord(path, payload)
    assert list(iter_tfrecord(path)) == [payload]
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError, match="truncated TFRecord payload"):
        list(iter_tfrecord(path))


def test_audit_counts_recording_groups_and_checks_pinned_shard_counts(tmp_path):
    lengths = [4] * 7 + [3] * 24
    episode_index = 0
    for index, length in enumerate(lengths):
        shard = tmp_path / f"r2d2_faceblur-train.tfrecord-{index:05d}-of-00031"
        with shard.open("wb") as stream:
            for _ in range(length):
                record_path = f"episode/{episode_index}"
                group = "recording/shared" if episode_index < 2 else f"recording/{episode_index}"
                payload = _episode_example(record_path, group)
                stream.write(struct.pack("<Q", len(payload)) + bytes(4) + payload + bytes(4))
                episode_index += 1
    info = {"splits": [{"name": "train", "shardLengths": [str(n) for n in lengths]}]}
    result = audit_dataset(tmp_path, info)
    assert result["episode_count"] == 100
    assert result["unique_source_files"] == 100
    assert result["episodes_per_recording_group"]["recording/shared"] == 2
    assert result["unique_recording_groups"] == 99
