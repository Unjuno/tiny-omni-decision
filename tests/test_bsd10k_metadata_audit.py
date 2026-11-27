from __future__ import annotations

import csv
import io
import json
import zipfile

import pytest

from scripts.audit_bsd10k_metadata import (
    audit_metadata,
    build_split_manifest,
    canonical_license,
)


def _row(sound_id: int, uploader: str, label: str, license_url: str, confidence: int = 4):
    return {
        "sound_id": str(sound_id),
        "class": label,
        "class_idx": "101",
        "class_top": "fx",
        "confidence": str(confidence),
        "uploader": uploader,
        "license": license_url,
        "title": f"private title {sound_id}",
        "tags": "private tags",
        "description": "private description",
    }


def _write_archive(path, rows) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("metadata/BSD10k_metadata.csv", buffer.getvalue())


def test_license_allowlist_requires_exact_versioned_media_license() -> None:
    assert canonical_license("https://creativecommons.org/licenses/by/4.0/") == "CC-BY-4.0"
    assert canonical_license("https://creativecommons.org/publicdomain/zero/1.0/") == "CC0-1.0"
    assert (
        canonical_license("https://creativecommons.org/licenses/by/3.0/") == "OTHER:licenses/by/3.0"
    )
    assert (
        canonical_license("https://creativecommons.org/licenses/by-nc/4.0/")
        == "OTHER:licenses/by-nc/4.0"
    )


def test_audit_excludes_disallowed_rows_and_keeps_uploader_groups_disjoint(tmp_path) -> None:
    rows = []
    for index in range(30):
        rows.append(
            _row(
                index,
                f"uploader-{index // 2}",
                ("fx-o", "ss-u", "sp-s")[index % 3],
                "https://creativecommons.org/licenses/by/4.0/",
                index % 5 + 1,
            )
        )
    rows.extend(
        [
            _row(30, "excluded-nc", "fx-o", "https://creativecommons.org/licenses/by-nc/4.0/"),
            _row(31, "excluded-old-by", "fx-o", "https://creativecommons.org/licenses/by/3.0/"),
        ]
    )
    archive = tmp_path / "metadata.zip"
    _write_archive(archive, rows)

    result = audit_metadata(archive)

    assert result["record_count"] == 32
    assert result["eligible_records"] == 30
    assert result["eligible_unique_uploader_groups"] == 15
    assert result["license_counts"]["OTHER:licenses/by-nc/4.0"] == 1
    assert result["license_counts"]["OTHER:licenses/by/3.0"] == 1
    assert result["group_split_preview"]["uploader_overlap_count"] == 0
    assert sum(result["group_split_preview"]["examples"].values()) == 30
    rendered = json.dumps(result)
    assert "private title" not in rendered
    assert "uploader-" not in rendered
    assert result["raw_identifiers_titles_and_urls_emitted"] is False

    split_manifest = build_split_manifest(archive)
    assigned_splits = {example["split"] for example in split_manifest["examples"]}
    assert assigned_splits == {"train", "validation", "test"}
    assert len({example["sound_id"] for example in split_manifest["examples"]}) == 30
    assert all("uploader_group_sha256" in example for example in split_manifest["examples"])
    assert "private title" not in json.dumps(split_manifest)
    assert "uploader-" not in json.dumps(split_manifest)
    group_splits = {}
    for example in split_manifest["examples"]:
        group_hash = example["uploader_group_sha256"]
        previous = group_splits.setdefault(group_hash, example["split"])
        assert previous == example["split"]


def test_audit_rejects_duplicate_ids_and_invalid_confidence(tmp_path) -> None:
    archive = tmp_path / "metadata.zip"
    duplicate = [
        _row(1, "uploader-a", "fx-o", "https://creativecommons.org/licenses/by/4.0/"),
        _row(1, "uploader-b", "fx-o", "https://creativecommons.org/licenses/by/4.0/"),
    ]
    _write_archive(archive, duplicate)
    with pytest.raises(ValueError, match="duplicate sound_id"):
        audit_metadata(archive)

    invalid = [_row(1, "uploader-a", "fx-o", "https://creativecommons.org/licenses/by/4.0/", 7)]
    _write_archive(archive, invalid)
    with pytest.raises(ValueError, match="1 through 5"):
        audit_metadata(archive)
