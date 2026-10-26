import hashlib

import pytest

from scripts.evaluate_openimages_pecore_zero_shot import _safe_image_path


def test_safe_image_path_checks_hash_and_stays_in_media_root(tmp_path):
    validation = tmp_path / "validation"
    validation.mkdir()
    image = validation / "image.jpg"
    image.write_bytes(b"image bytes")
    digest = hashlib.sha256(b"image bytes").hexdigest()
    row = {
        "local_path": "image.jpg",
        "media_sha256": digest,
        "_artifact_root": str(tmp_path),
    }
    assert _safe_image_path(row) == image.resolve()


def test_safe_image_path_rejects_tampered_media(tmp_path):
    validation = tmp_path / "validation"
    validation.mkdir()
    (validation / "image.jpg").write_bytes(b"changed bytes")
    row = {
        "local_path": "image.jpg",
        "media_sha256": "0" * 64,
        "_artifact_root": str(tmp_path),
    }
    with pytest.raises(ValueError, match="media SHA-256 mismatch"):
        _safe_image_path(row)


def test_safe_image_path_rejects_escape_paths(tmp_path):
    validation = tmp_path / "validation"
    validation.mkdir()
    row = {"local_path": "../outside.jpg", "_artifact_root": str(tmp_path)}
    with pytest.raises(ValueError, match="safe relative path"):
        _safe_image_path(row)
