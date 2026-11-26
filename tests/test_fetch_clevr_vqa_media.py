import hashlib
import zipfile

import pytest


def test_validation_manifest_maps_to_official_val_archive_directory():
    from scripts.fetch_clevr_vqa_media import archive_member

    assert archive_member("train", "CLEVR_train_000001.png") == (
        "CLEVR_v1.0/images/train/CLEVR_train_000001.png"
    )
    assert archive_member("validation", "CLEVR_val_000001.png") == (
        "CLEVR_v1.0/images/val/CLEVR_val_000001.png"
    )


def test_extract_member_checks_local_header_crc_and_hash(tmp_path, monkeypatch):
    from scripts import fetch_clevr_vqa_media as fetcher

    archive_path = tmp_path / "images.zip"
    name = "CLEVR_v1.0/images/train/CLEVR_train_000001.png"
    original = b"tiny deterministic png-like bytes" * 100
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(name, original)
    with zipfile.ZipFile(archive_path) as archive:
        info = archive.getinfo(name)
        payload_bytes = archive_path.read_bytes()
        monkeypatch.setattr(
            fetcher,
            "fetch_range",
            lambda start, end: payload_bytes[start : end + 1],
        )
        restored, digest = fetcher.extract_member(info)
    assert restored == original
    assert digest == hashlib.sha256(original).hexdigest()


def test_extract_member_rejects_mismatched_local_name(tmp_path, monkeypatch):
    from scripts import fetch_clevr_vqa_media as fetcher

    archive_path = tmp_path / "images.zip"
    name = "CLEVR_v1.0/images/train/CLEVR_train_000001.png"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(name, b"payload")
    with zipfile.ZipFile(archive_path) as archive:
        info = archive.getinfo(name)
        payload_bytes = bytearray(archive_path.read_bytes())
        payload_bytes[info.header_offset + 30] ^= 1
        monkeypatch.setattr(
            fetcher,
            "fetch_range",
            lambda start, end: bytes(payload_bytes[start : end + 1]),
        )
        with pytest.raises(ValueError, match="names differ"):
            fetcher.extract_member(info)
