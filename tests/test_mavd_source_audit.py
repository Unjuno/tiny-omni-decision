from __future__ import annotations

import zipfile
from pathlib import Path

from scripts import audit_mavd_traffic_source as audit


def _zip(path: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
        for name, payload in members.items():
            zipped.writestr(name, payload)


def test_audit_keeps_test_annotation_payload_unread(tmp_path, monkeypatch):
    _zip(
        tmp_path / "annotations_train.zip",
        {"annotations_train/rivera0101_00.txt": b"0.0\t1.0\tcar\n"},
    )
    _zip(
        tmp_path / "annotations_validate.zip",
        {"annotations_validate/rivera0101_01.txt": b"0.0\t1.0\tbus\n"},
    )
    _zip(
        tmp_path / "annotations_test.zip",
        {"annotations_test/acevedo0101_00.txt": b"invalid utf-8: \xff\n"},
    )
    (tmp_path / "metadata.txt").write_text(
        "file,latitude,longitude,year,month,day,hour,minute\n"
        "rivera0101_00.wav,0,0,2020,01,01,12,00\n"
        "rivera0101_01.wav,0,0,2020,01,01,12,05\n"
        "acevedo0101_00.wav,1,1,2020,01,01,13,00\n",
        encoding="utf-8",
    )
    (tmp_path / "README").write_text("test bundle\n", encoding="utf-8")
    monkeypatch.setattr(
        audit,
        "EXPECTED_MD5",
        {name: audit.file_hashes(tmp_path / name)["md5"] for name in audit.EXPECTED_MD5},
    )

    result = audit.audit_source(tmp_path)

    assert result["cross_split_site_day_groups"] == {"rivera:2020-01-01": ["train", "validate"]}
    assert result["development_label_interval_counts"] == {"train": 1, "validate": 1}
    assert result["test_member_names_only"] == ["acevedo0101_00"]
    assert result["test_annotation_payloads_read"] is False
    assert result["media_payloads_downloaded_or_opened"] is False


def test_test_member_listing_does_not_decode_zip_payload(tmp_path):
    archive = tmp_path / "test.zip"
    _zip(archive, {"annotations_test/scene_00.txt": b"\xff\xfe\x00"})

    assert audit._member_stems(archive) == ["scene_00"]


def test_development_annotation_parser_rejects_invalid_interval(tmp_path):
    archive = tmp_path / "train.zip"
    _zip(
        archive,
        {"annotations_train/scene_00.txt": b"2.0\t1.0\tcar\n0\t1\tbus\n"},
    )

    labels, count, invalid = audit._read_development_labels(archive)

    assert labels == {"bus": 1}
    assert count == 1
    assert invalid == ["annotations_train/scene_00.txt:1:invalid_interval"]
