from __future__ import annotations

import numpy as np
import pytest

from scripts.extract_physical_concepts_probe_features import (
    ROW_BYTES,
    _preprocess_probe_frames,
    _validate_probe_row,
    media_identity_overlaps,
    sample_rank,
    select_probe_records,
)


def test_probe_sample_hash_rank_is_deterministic_and_category_scoped() -> None:
    first = sample_rank("continuity", b"record-1")
    assert first == sample_rank("continuity", b"record-1")
    assert first != sample_rank("solidity", b"record-1")
    assert first[1] != sample_rank("continuity", b"record-2")[1]


def test_probe_selection_takes_smallest_hash_ranks() -> None:
    records = [
        ("f" * 64, "row-c", "shard", 2),
        ("1" * 64, "row-a", "shard", 0),
        ("8" * 64, "row-b", "shard", 1),
    ]
    assert select_probe_records(records, 2) == [records[1], records[2]]


def test_probe_selection_rejects_duplicate_rows_and_short_population() -> None:
    duplicated = [("a" * 64, "row", "shard-a", 0), ("b" * 64, "row", "shard-b", 0)]
    with pytest.raises(ValueError, match="duplicate serialized"):
        select_probe_records(duplicated, 1)
    with pytest.raises(ValueError, match="quota"):
        select_probe_records([("a" * 64, "row", "shard", 0)], 2)


def test_selected_rgb_media_identity_is_checked_against_both_splits() -> None:
    overlaps = media_identity_overlaps(
        ["a", "b", "c", "a"],
        {"train": {"a", "z"}, "validation": {"c"}},
    )
    assert overlaps == {"train": {"a"}, "validation": {"c"}}


def test_probe_schema_decodes_four_clips_with_ordered_binary_labels(monkeypatch) -> None:
    from scripts import audit_physical_concepts_tfrecord

    features = {}
    for name in ("possible", "impossible"):
        features[f"{name}_image"] = ("bytes", [bytes(2 * ROW_BYTES)])
        features[f"{name}_mask"] = ("bytes", [bytes(2 * 15 * 64 * 64)])
        features[f"{name}_camera_pose"] = ("float", [0.0] * 180)
    monkeypatch.setattr(audit_physical_concepts_tfrecord, "_feature_map", lambda _: features)

    videos, labels, rgb_hashes = _validate_probe_row(b"test", "continuity", np)

    assert videos.shape == (4, 15, 64, 64, 3)
    assert labels == [0, 0, 1, 1]
    assert len(rgb_hashes) == 4


def test_probe_schema_rejects_missing_fields(monkeypatch) -> None:
    from scripts import audit_physical_concepts_tfrecord

    monkeypatch.setattr(audit_physical_concepts_tfrecord, "_feature_map", lambda _: {})
    with pytest.raises(ValueError, match="unexpected continuity probe keys"):
        _validate_probe_row(b"test", "continuity", np)


def test_probe_preprocessing_normalizes_read_only_tensor_array_out_of_place() -> None:
    class ImageOps:
        @staticmethod
        def resize(frames, _size, *, method):
            assert method == "bilinear"
            return np.frombuffer(
                np.ones((len(frames), 288, 288, 3), dtype=np.float32).tobytes(),
                dtype=np.float32,
            ).reshape(len(frames), 288, 288, 3)

        @staticmethod
        def resize_with_crop_or_pad(frames, _size, _width=None):
            return frames

    class FakeTensorFlow:
        image = ImageOps()

    frames = np.zeros((4, 8, 64, 64, 3), dtype=np.uint8)
    result = _preprocess_probe_frames(frames, FakeTensorFlow(), np)

    assert result.shape == (4, 8, 288, 288, 3)
    assert np.all(result == np.float32(1 / 255))
