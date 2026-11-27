from __future__ import annotations

import csv
import json
import zipfile

import pytest

from scripts.audit_soundactions_metadata import audit_metadata


def _write_csv(path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _overview_row(number: str) -> dict[str, str]:
    return {
        "number": number,
        "title": f"private title {number}",
        "duration_s": "4.0",
        "perception_type": "impulsive",
        "wav": f"SoundAction {int(number):03d} - title.wav",
        "mp4": f"SoundAction {int(number):03d} - title.mp4",
        "jpg": f"SoundAction {int(number):03d} - title.jpg",
    }


def test_audit_reports_group_and_label_quality_without_raw_titles(tmp_path) -> None:
    overview = tmp_path / "overview.csv"
    _write_csv(overview, [_overview_row("1"), _overview_row("2"), _overview_row("3")])
    labels = tmp_path / "labels.zip"
    with zipfile.ZipFile(labels, "w") as archive:
        for rater, values in zip("ABC", [("impulse", "sustain", "iterative")] * 3, strict=True):
            rows = [
                {
                    "SoundAction #": f"SoundAction {index:03d}",
                    "Schaffer mass type": value,
                }
                for index, value in enumerate(values, 1)
            ]
            import io

            buffer = io.StringIO()
            writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            archive.writestr(f"Labels/SoundActions_labeling_{rater}.csv", buffer.getvalue())

        master_rows = [
            {
                "number": f"{index:03d}",
                "perception_type": label,
                "ptype_n_agree": "3",
                "actions": f"action {index}",
                "objects": f"object {index}",
            }
            for index, label in enumerate(("impulsive", "sustained", "iterative"), 1)
        ]
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=list(master_rows[0]))
        writer.writeheader()
        writer.writerows(master_rows)
        archive.writestr("Labels/SoundActions_labels_master.csv", buffer.getvalue())
        archive.writestr("label-corrections.csv", "number,title,column,deposited,corrected\n")

    result = audit_metadata(overview, labels)

    assert result["record_count"] == 3
    assert result["annotation_rater_count"] == 3
    assert result["split_column_in_overview"] is False
    assert result["media_references"]["mp4"]["unique_references"] == 3
    assert result["label_agreement"]["perception_type"]["unanimous_items"] == 3
    assert result["label_agreement"]["perception_type"]["master_consensus_matches_majority"] == 3
    assert result["raw_titles_urls_and_ids_emitted"] is False
    assert "private title" not in json.dumps(result)


def test_audit_rejects_duplicate_record_ids(tmp_path) -> None:
    overview = tmp_path / "overview.csv"
    _write_csv(overview, [_overview_row("1"), _overview_row("1")])
    labels = tmp_path / "labels.zip"
    with zipfile.ZipFile(labels, "w"):
        pass

    with pytest.raises(ValueError, match="duplicate recording IDs"):
        audit_metadata(overview, labels)
