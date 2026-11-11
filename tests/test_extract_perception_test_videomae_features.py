from __future__ import annotations

import json

import numpy as np
import pytest

from scripts.extract_perception_test_videomae_features import (
    FRAME_COUNT,
    FRAME_SIZE,
    TOKEN_SHAPE,
    _load_preflight_progress,
    _receipt_set_hash,
    preprocess_frames,
    sha256_file,
    uniform_indices,
)


def test_uniform_frame_indices_keep_endpoints_and_fixed_count():
    values = uniform_indices(651)
    assert len(values) == FRAME_COUNT
    assert values[0] == 0
    assert values[-1] == 650
    assert values == sorted(set(values))
    assert TOKEN_SHAPE == (8, 14, 14, 384)


def test_uniform_frame_indices_reject_short_videos():
    with pytest.raises(ValueError, match="at least"):
        uniform_indices(FRAME_COUNT - 1)


def test_preprocess_frames_matches_fixed_float_range():
    frames = np.zeros((FRAME_COUNT, FRAME_SIZE, FRAME_SIZE, 3), dtype=np.uint8)
    frames[-1] = 255
    result = preprocess_frames(frames)
    assert result.shape == frames.shape
    assert result.dtype == np.float32
    assert result[0].min() == -1.0
    assert result[-1].max() == 1.0
    assert np.isfinite(result).all()


def test_preprocess_frames_rejects_wrong_shape_or_dtype():
    with pytest.raises(ValueError, match="pinned RGB"):
        preprocess_frames(np.zeros((2, 224, 224, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="pinned RGB"):
        preprocess_frames(np.zeros((FRAME_COUNT, FRAME_SIZE, FRAME_SIZE, 3), dtype=np.float32))


def test_complete_train_progress_receipts_are_stable_when_validation_appends(tmp_path, monkeypatch):
    import scripts.extract_perception_test_videomae_features as extractor

    members = [
        {
            "split": "train",
            "video_id": f"video_{index}",
            "member": f"videos/video_{index}.mp4",
            "expanded_bytes": 100 + index,
            "compressed_bytes": 50 + index,
            "crc32": f"{index:08x}",
        }
        for index in range(2)
    ]
    preflight_path = tmp_path / "preflight.json"
    preflight_path.write_text(
        json.dumps(
            {
                "status": "PASS_METADATA_ONLY_MEDIA_PREFLIGHT",
                "official_test_archive_opened": False,
                "splits": {"train": {"members": members}},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(extractor, "PREFLIGHT_SHA256", sha256_file(preflight_path))
    progress_path = tmp_path / "progress.jsonl"
    train_receipts = [
        {
            **member,
            "local_relative_path": f"train/{member['video_id']}.mp4",
            "sha256": f"{index:064x}",
            "crc32_verified": True,
            "expanded_size_verified": True,
        }
        for index, member in enumerate(members)
    ]
    progress_path.write_text(
        "".join(json.dumps(row) + "\n" for row in train_receipts),
        encoding="utf-8",
    )

    preflight_hash, receipt_hash, progress_hash, loaded = _load_preflight_progress(
        preflight_path, progress_path, "train"
    )
    assert preflight_hash == sha256_file(preflight_path)
    assert receipt_hash == _receipt_set_hash(train_receipts)
    assert progress_hash == sha256_file(progress_path)
    assert loaded == sorted(train_receipts, key=lambda row: row["video_id"])

    progress_path.write_text(
        progress_path.read_text(encoding="utf-8")
        + json.dumps(
            {
                "split": "valid",
                "video_id": "validation-only",
                "member": "videos/validation-only.mp4",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _, receipt_hash_after_valid, _, _ = _load_preflight_progress(
        preflight_path, progress_path, "train"
    )
    assert receipt_hash_after_valid == receipt_hash


def test_preflight_progress_rejects_incomplete_or_duplicate_media_receipts(tmp_path, monkeypatch):
    import scripts.extract_perception_test_videomae_features as extractor

    preflight = {
        "status": "PASS_METADATA_ONLY_MEDIA_PREFLIGHT",
        "official_test_archive_opened": False,
        "splits": {
            "train": {
                "members": [
                    {
                        "split": "train",
                        "video_id": "video_1",
                        "member": "videos/video_1.mp4",
                        "expanded_bytes": 100,
                        "compressed_bytes": 50,
                        "crc32": "00000001",
                    }
                ]
            }
        },
    }
    preflight_path = tmp_path / "preflight.json"
    preflight_path.write_text(json.dumps(preflight), encoding="utf-8")
    monkeypatch.setattr(extractor, "PREFLIGHT_SHA256", sha256_file(preflight_path))
    progress_path = tmp_path / "progress.jsonl"
    progress_path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="incomplete"):
        _load_preflight_progress(preflight_path, progress_path, "train")

    receipt = {
        "split": "train",
        "video_id": "video_1",
        "member": "videos/video_1.mp4",
        "expanded_bytes": 100,
        "compressed_bytes": 50,
        "crc32": "00000001",
        "local_relative_path": "train/video_1.mp4",
        "sha256": "a" * 64,
        "crc32_verified": True,
        "expanded_size_verified": True,
    }
    progress_path.write_text(
        json.dumps(receipt) + "\n" + json.dumps(receipt) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="duplicate"):
        _load_preflight_progress(preflight_path, progress_path, "train")
