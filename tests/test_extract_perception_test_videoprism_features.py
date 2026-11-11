from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.extract_perception_test_videoprism_features import (
    _load_fixed_inputs,
    uniform_indices,
)


def test_uniform_indices_are_endpoint_inclusive_and_unique():
    assert uniform_indices(8) == list(range(8))
    assert uniform_indices(15, 4) == [0, 5, 9, 14]


def test_uniform_indices_reject_short_or_duplicate_sampling():
    with pytest.raises(ValueError, match="at least"):
        uniform_indices(7, 8)
    assert uniform_indices(12, 1) == [0]


def _fixed_media_files():
    return [
        {
            "video_id": "video_1",
            "split": "train",
            "member": "videos/video_1.mp4",
            "expanded_bytes": 10,
            "compressed_bytes": 8,
            "crc32": "00000001",
            "local_relative_path": "train/video_1.mp4",
            "sha256": "1" * 64,
        },
        {
            "video_id": "video_2",
            "split": "valid",
            "member": "videos/video_2.mp4",
            "expanded_bytes": 11,
            "compressed_bytes": 9,
            "crc32": "00000002",
            "local_relative_path": "valid/video_2.mp4",
            "sha256": "2" * 64,
        },
    ]


def test_load_fixed_inputs_matches_split_ids_and_media_identity(tmp_path):
    preflight = {
        "status": "PASS_METADATA_ONLY_MEDIA_PREFLIGHT",
        "official_test_archive_opened": False,
        "splits": {
            "train": {"members": [_fixed_media_files()[0]]},
            "valid": {"members": [_fixed_media_files()[1]]},
        },
    }
    preflight_path = tmp_path / "preflight.json"
    preflight_path.write_text(json.dumps(preflight), encoding="utf-8")
    media_manifest = {
        "media_preflight_sha256": _sha256(preflight_path),
        "official_test_archive_opened": False,
        "media_files": _fixed_media_files(),
    }
    manifest_path = tmp_path / "media-manifest.json"
    manifest_path.write_text(json.dumps(media_manifest), encoding="utf-8")

    preflight_hash, manifest_hash, rows = _load_fixed_inputs(
        preflight_path, manifest_path, "train"
    )

    assert preflight_hash == _sha256(preflight_path)
    assert manifest_hash == _sha256(manifest_path)
    assert [row["video_id"] for row in rows] == ["video_1"]


def test_load_fixed_inputs_rejects_split_or_media_mismatch(tmp_path):
    preflight = {
        "status": "PASS_METADATA_ONLY_MEDIA_PREFLIGHT",
        "official_test_archive_opened": False,
        "splits": {"train": {"members": [_fixed_media_files()[0]]}},
    }
    preflight_path = tmp_path / "preflight.json"
    preflight_path.write_text(json.dumps(preflight), encoding="utf-8")
    media_manifest = {
        "media_preflight_sha256": _sha256(preflight_path),
        "official_test_archive_opened": False,
        "media_files": [],
    }
    manifest_path = tmp_path / "media-manifest.json"
    manifest_path.write_text(json.dumps(media_manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="media IDs differ"):
        _load_fixed_inputs(preflight_path, manifest_path, "train")


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()
