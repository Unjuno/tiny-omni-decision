from __future__ import annotations

import csv
import hashlib
import io
import json
import zipfile

import pytest

from scripts import prepare_bsd10k_ccby3_train_extension as extension


def _uploader_hash(name: str) -> str:
    return hashlib.sha256(name.strip().casefold().encode()).hexdigest()


def _fixture(tmp_path, monkeypatch):
    metadata_rows = [
        {
            "sound_id": "1",
            "class": "fx-a",
            "confidence": "4",
            "uploader": "train-a",
            "license": "https://creativecommons.org/licenses/by/4.0/",
            "title": "base",
        },
        {
            "sound_id": "6",
            "class": "fx-a",
            "confidence": "4",
            "uploader": "train-z",
            "license": "https://creativecommons.org/licenses/by/4.0/",
            "title": "base z",
        },
        {
            "sound_id": "2",
            "class": "fx-a",
            "confidence": "4",
            "uploader": "heldout",
            "license": "https://creativecommons.org/licenses/by/4.0/",
            "title": "heldout",
        },
        {
            "sound_id": "3",
            "class": "fx-a",
            "confidence": "5",
            "uploader": "new-b",
            "license": "https://creativecommons.org/licenses/by/3.0/",
            "title": "new b",
        },
        {
            "sound_id": "4",
            "class": "fx-h",
            "confidence": "3",
            "uploader": "train-a",
            "license": "https://creativecommons.org/licenses/by/3.0/",
            "title": "new a",
        },
        {
            "sound_id": "5",
            "class": "fx-a",
            "confidence": "2",
            "uploader": "heldout",
            "license": "https://creativecommons.org/licenses/by/3.0/",
            "title": "must exclude",
        },
    ]
    metadata = io.StringIO(newline="")
    writer = csv.DictWriter(metadata, fieldnames=list(metadata_rows[0]))
    writer.writeheader()
    writer.writerows(metadata_rows)
    metadata_zip = tmp_path / "metadata.zip"
    with zipfile.ZipFile(metadata_zip, "w") as archive:
        archive.writestr("BSD10k_metadata.csv", metadata.getvalue())

    base = {
        "schema": "tiny-omni-decision/bsd10k-group-split-v1",
        "source_record_id": "19868804",
        "metadata_zip_sha256": extension.sha256_file(metadata_zip),
        "examples": [
            {
                "sound_id": "1",
                "class": "fx-a",
                "confidence": 4,
                "license": "CC-BY-4.0",
                "uploader_group_sha256": _uploader_hash("train-a"),
                "split": "train",
            },
            {
                "sound_id": "6",
                "class": "fx-a",
                "confidence": 4,
                "license": "CC-BY-4.0",
                "uploader_group_sha256": _uploader_hash("train-z"),
                "split": "train",
            },
            {
                "sound_id": "2",
                "class": "fx-a",
                "confidence": 4,
                "license": "CC-BY-4.0",
                "uploader_group_sha256": _uploader_hash("heldout"),
                "split": "validation",
            },
        ],
    }
    split = tmp_path / "base.json"
    split.write_text(json.dumps(base), encoding="utf-8")
    monkeypatch.setattr(extension, "EXPECTED_METADATA_SHA256", extension.sha256_file(metadata_zip))
    monkeypatch.setattr(extension, "EXPECTED_BASE_SPLIT_SHA256", extension.sha256_file(split))
    monkeypatch.setattr(extension, "EXPECTED_CC_BY3_COUNT", 3)
    monkeypatch.setattr(extension, "EXPECTED_HELDOUT_GROUP_EXCLUSIONS", 1)
    monkeypatch.setattr(extension, "EXPECTED_EXTENSION_COUNT", 2)
    monkeypatch.setattr(extension, "EXPECTED_EXTENSION_GROUPS", 2)
    monkeypatch.setattr(extension, "EXPECTED_NEW_GROUPS", 1)
    return metadata_zip, split, base


def test_extension_preserves_heldout_and_emits_private_attribution(tmp_path, monkeypatch):
    metadata_zip, base_path, base = _fixture(tmp_path, monkeypatch)
    combined_path, additions_path = tmp_path / "combined.json", tmp_path / "additions.json"
    roles_path = tmp_path / "group-roles.json"
    attribution_path, report_path = tmp_path / "attribution.csv", tmp_path / "report.json"

    report = extension.prepare_extension(
        metadata_zip,
        base_path,
        combined_path,
        additions_path,
        roles_path,
        attribution_path,
        report_path,
    )

    combined = json.loads(combined_path.read_text(encoding="utf-8"))
    additions = json.loads(additions_path.read_text(encoding="utf-8"))
    old_heldout = [row for row in base["examples"] if row["split"] != "train"]
    new_heldout = [row for row in combined["examples"] if row["split"] != "train"]
    assert new_heldout == old_heldout
    assert [row["sound_id"] for row in additions["examples"]] == ["3", "4"]
    assert all(row["split"] == "train" for row in additions["examples"])
    assert report["validation_and_test_assignments_changed"] is False
    with attribution_path.open(encoding="utf-8", newline="") as stream:
        attribution = list(csv.DictReader(stream))
    assert {row["creator_uploader"] for row in attribution} == {"new-b", "train-a"}
    assert not report["custom_test_or_sealed_audit_loaded"]
    roles = json.loads(roles_path.read_text(encoding="utf-8"))
    assert len(roles["calibration_uploader_group_sha256"]) == 1


def test_extension_refuses_to_overwrite_any_output(tmp_path, monkeypatch):
    metadata_zip, base_path, _ = _fixture(tmp_path, monkeypatch)
    occupied = tmp_path / "occupied"
    occupied.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        extension.prepare_extension(
            metadata_zip,
            base_path,
            occupied,
            tmp_path / "additions.json",
            tmp_path / "roles.json",
            tmp_path / "attribution.csv",
            tmp_path / "report.json",
        )
    assert occupied.read_text(encoding="utf-8") == "keep"
