import hashlib
import zipfile

import pytest


def test_archive_member_maps_only_valb_image_basenames():
    from scripts.fetch_clevr_cogent_b_audit_media import archive_member

    assert archive_member("CLEVR_valB_000001.png") == (
        "CLEVR_CoGenT_v1.0/images/valB/CLEVR_valB_000001.png"
    )
    for invalid in ("CLEVR_train_000001.png", "../CLEVR_valB_000001.png"):
        with pytest.raises(ValueError):
            archive_member(invalid)


def test_extract_member_checks_zip_header_crc_and_payload_sha(tmp_path, monkeypatch):
    from scripts import fetch_clevr_cogent_b_audit_media as fetcher

    path = tmp_path / "source.zip"
    name = "CLEVR_CoGenT_v1.0/images/valB/CLEVR_valB_000001.png"
    payload = b"valid fixture image bytes" * 100
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(name, payload)
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo(name)
        archive_bytes = path.read_bytes()
        monkeypatch.setattr(
            fetcher, "fetch_range", lambda start, end: archive_bytes[start : end + 1]
        )
        restored, digest = fetcher.extract_member(info)
    assert restored == payload
    assert digest == hashlib.sha256(payload).hexdigest()


def test_extract_member_rejects_local_header_name_mismatch(tmp_path, monkeypatch):
    from scripts import fetch_clevr_cogent_b_audit_media as fetcher

    path = tmp_path / "source.zip"
    name = "CLEVR_CoGenT_v1.0/images/valB/CLEVR_valB_000001.png"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(name, b"payload")
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo(name)
        archive_bytes = bytearray(path.read_bytes())
        archive_bytes[info.header_offset + 30] ^= 1
        monkeypatch.setattr(
            fetcher,
            "fetch_range",
            lambda start, end: bytes(archive_bytes[start : end + 1]),
        )
        with pytest.raises(ValueError, match="names differ"):
            fetcher.extract_member(info)
