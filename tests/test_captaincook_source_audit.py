import csv
import json
from pathlib import Path

from scripts.audit_captaincook4d_source import audit_source


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def test_audit_ignores_test_rows_and_reports_train_validation_only(tmp_path: Path) -> None:
    root = tmp_path / "annotations"
    (root / "data_splits").mkdir(parents=True)
    (root / "data_splits" / "person_data_split_combined.json").write_text(
        json.dumps({"train": ["tr-1"], "val": ["va-1"], "test": ["te-1"]}),
        encoding="utf-8",
    )
    _write_csv(
        root / "metadata" / "video_information.csv",
        ["recording_id", "environment_id", "person_id"],
        [["tr-1", "env-1", "person-1"], ["va-1", "env-2", "person-2"]],
    )
    _write_csv(
        root / "annotation_csv" / "recording_id_step_idx.csv",
        ["recording_id", "activity_id"],
        [["tr-1", "act-1"], ["va-1", "act-1"]],
    )
    _write_csv(
        root / "annotation_csv" / "step_annotations.csv",
        ["recording_id", "has_errors"],
        [["tr-1", "false"], ["va-1", "true"], ["te-1", "not-a-label"]],
    )
    links = tmp_path / "download_links.json"
    links.write_text(
        json.dumps(
            {
                "tr-1": {"gopro_360p": "url"},
                "va-1": {"gopro_360p": "url"},
            }
        ),
        encoding="utf-8",
    )

    report = audit_source(root, links)

    assert report["official_test_split_metrics_loaded"] is False
    assert report["split_audit"]["person_overlap"] == 0
    assert report["coverage"]["train"]["negative_steps"] == 1
    assert report["coverage"]["validation"]["positive_steps"] == 1
    assert "test" not in report["coverage"]
    assert report["task_disposition"]["audio_video_complementarity"] == "NOT_ESTABLISHED"


def test_audit_detects_person_overlap_and_official_split_contamination(tmp_path: Path) -> None:
    root = tmp_path / "annotations"
    (root / "data_splits").mkdir(parents=True)
    (root / "data_splits" / "person_data_split_combined.json").write_text(
        json.dumps({"train": ["tr-1", "tr-2"], "val": ["va-1"], "test": []}),
        encoding="utf-8",
    )
    _write_csv(
        root / "metadata" / "video_information.csv",
        ["recording_id", "environment_id", "person_id"],
        [
            ["tr-1", "env-1", "shared"],
            ["tr-2", "env-1", "train-only"],
            ["va-1", "env-2", "shared"],
        ],
    )
    _write_csv(
        root / "annotation_csv" / "recording_id_step_idx.csv",
        ["recording_id", "activity_id"],
        [["tr-1", "act-1"], ["tr-2", "act-1"], ["va-1", "act-2"]],
    )
    _write_csv(
        root / "annotation_csv" / "step_annotations.csv",
        ["recording_id", "has_errors"],
        [["tr-1", "true"], ["tr-2", "false"], ["va-1", "true"]],
    )
    links = tmp_path / "download_links.json"
    links.write_text(
        json.dumps(
            {
                "tr-1": {"gopro_360p": "url"},
                "tr-2": {"gopro_360p": "url"},
                "va-1": {"gopro_360p": "url"},
            }
        ),
        encoding="utf-8",
    )

    report = audit_source(root, links)

    assert report["split_audit"]["person_overlap"] == 1
    assert report["split_audit"]["shared_person_train_recordings"] == ["tr-1"]
    assert report["split_audit"]["train_after_removing_shared_person_recordings"] == 1
    assert report["split_audit"]["person_overlap_after_removal"] == 0
    assert report["split_audit"]["activity_overlap"] == 0
