from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest
from PIL import Image

from scripts import audit_intphys_train_archive_structure as audit


def _write_fixture(
    path: Path, *, omit_last_frame: bool = False, impossible_label: bool = False
) -> None:
    def png_bytes(color: tuple[int, int, int]) -> bytes:
        buffer = io.BytesIO()
        Image.new("RGB", (2, 2), color).save(buffer, format="PNG")
        return buffer.getvalue()

    with tarfile.open(path, "w:gz") as archive:
        for video_id in ("video-a", "video-b"):
            status = {
                "header": {"is_possible": not impossible_label},
                "frames": [{} for _ in range(audit.EXPECTED_FRAMES)],
            }
            payload = json.dumps(status).encode()
            info = tarfile.TarInfo(f"train/{video_id}/status.json")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
            for channel in ("scene", "depth", "masks"):
                frame_limit = audit.EXPECTED_FRAMES - int(
                    omit_last_frame and video_id == "video-b" and channel == "scene"
                )
                for frame in range(1, frame_limit + 1):
                    info = tarfile.TarInfo(f"train/{video_id}/{channel}/{channel}_{frame:03d}.png")
                    pixels = png_bytes((frame, 0, 0)) if channel == "scene" else b"fixture"
                    info.size = len(pixels)
                    archive.addfile(info, io.BytesIO(pixels))


def _patch_archive_pin(monkeypatch: pytest.MonkeyPatch, archive: Path) -> None:
    content = archive.read_bytes()
    monkeypatch.setitem(
        audit.TRAIN_ARCHIVE_PINS,
        2,
        {
            "bytes": len(content),
            "md5": hashlib.md5(content, usedforsecurity=False).hexdigest(),
            "url": "https://fixture.invalid/train.2.tar.gz",
        },
    )


def test_archive_audit_verifies_pin_and_complete_train_structure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / "train.2.tar.gz"
    _write_fixture(archive)
    _patch_archive_pin(monkeypatch, archive)

    result = audit.audit_train_archive(archive, shard_number=2, expected_video_count=2)

    assert result["status"] == "pass_train_shard_structure_audit"
    assert result["publisher_pin_verified"] is True
    assert result["videos"] == 2
    assert result["scene_frames"] == result["depth_frames"] == result["mask_frames"] == 200
    assert result["unique_rgb_sequence_hashes"] == 1
    assert result["unique_sampled_rgb_payload_hashes"] == 1
    assert result["unique_sampled_decoded_rgb_pixel_hashes"] == 1
    identities = result["identity_records"]
    assert [row["sample_id"] for row in identities] == [
        "train.2/video-a",
        "train.2/video-b",
    ]
    assert all(len(row["rgb_sequence_sha256"]) == 64 for row in identities)
    assert all(len(row["sampled_decoded_rgb_pixels_sha256"]) == 64 for row in identities)
    assert result["validation_or_sealed_data_accessed"] is False
    assert result["model_or_gpu_used"] is False


def test_archive_audit_rejects_incomplete_video_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / "train.2.tar.gz"
    _write_fixture(archive, omit_last_frame=True)
    _patch_archive_pin(monkeypatch, archive)

    with pytest.raises(ValueError, match="missing/incomplete"):
        audit.audit_train_archive(archive, shard_number=2, expected_video_count=2)


def test_archive_audit_rejects_non_possible_training_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / "train.2.tar.gz"
    _write_fixture(archive, impossible_label=True)
    _patch_archive_pin(monkeypatch, archive)

    with pytest.raises(ValueError, match="every sample possible"):
        audit.audit_train_archive(archive, shard_number=2, expected_video_count=2)


def test_archive_audit_rejects_archive_hash_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / "train.2.tar.gz"
    _write_fixture(archive)
    _patch_archive_pin(monkeypatch, archive)
    archive.write_bytes(archive.read_bytes() + b"changed")

    with pytest.raises(ValueError, match="archive size mismatch"):
        audit.audit_train_archive(archive, shard_number=2, expected_video_count=2)
