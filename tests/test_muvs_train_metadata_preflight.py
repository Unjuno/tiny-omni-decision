from __future__ import annotations

import csv
import hashlib
import io
import zipfile

import pytest

from tiny_omni_decision.muvs_preflight import inspect_muvs_train_archive


def _csv_bytes(header: list[str], rows: list[list[str]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(header)
    writer.writerows(rows)
    return output.getvalue().encode()


def _fixture_zip(path, *, selected_camera="cam_1"):
    fragment_header = [
        "event_id",
        "camera_id",
        "period",
        "ActiveCameras",
        "fragment_id",
        "Accelerometer_x",
        "Accelerometer_y",
        "Accelerometer_z",
        "Gyroscope_x",
        "Gyroscope_y",
        "Gyroscope_z",
    ]
    annotation_header = [
        "dataset_type",
        "event",
        "period",
        "fragment_number",
        "camera_selected",
        "offsets",
    ]
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "data/fragments/train/event_a_FRAGMENTS.csv",
            _csv_bytes(
                fragment_header,
                [
                    ["event_a", "cam_1", "1", "2", "1_cam_phone1", "0", "0", "1", "0", "0", "0"],
                    ["event_a", "cam_2", "1", "2", "1_cam_phone2", "0", "0", "1", "1", "0", "0"],
                ],
            ),
        )
        archive.writestr(
            "data/annotations/event_a_ANNOTATIONS.csv",
            _csv_bytes(
                annotation_header,
                [["train", "event_a_filtered", "1", "1", selected_camera, "{}"]],
            ),
        )
        # These entries exist but must never be read by the preflight.
        archive.writestr(
            "data/fragments/test/event_b_FRAGMENTS.csv", b"not a readable CSV"
        )
        archive.writestr(
            "data/annotations/event_b_ANNOTATIONS.csv", b"not a readable CSV"
        )


def test_reads_train_payloads_only_and_validates_choice_set(tmp_path):
    archive = tmp_path / "muvs.zip"
    _fixture_zip(archive)
    expected_md5 = hashlib.md5(archive.read_bytes(), usedforsecurity=False).hexdigest()

    report = inspect_muvs_train_archive(archive, expected_md5=expected_md5)

    assert report["source"]["train_event_count_from_fragment_files"] == 1
    assert report["source"]["test_event_count_from_fragment_filenames_only"] == 1
    assert report["source"]["test_fragment_member_contents_decompressed"] is False
    assert report["source"]["test_annotation_member_contents_decompressed"] is False
    assert report["source"]["test_partition_labels_or_sensor_values_parsed"] is False
    assert report["task"]["train_decision_count"] == 1
    assert report["task"]["choice_count_distribution"] == {2: 1}
    assert report["task"]["selected_camera_counts"] == {"cam_1": 1}


def test_rejects_target_absent_from_candidate_options(tmp_path):
    archive = tmp_path / "muvs.zip"
    _fixture_zip(archive, selected_camera="cam_3")
    expected_md5 = hashlib.md5(archive.read_bytes(), usedforsecurity=False).hexdigest()

    with pytest.raises(ValueError, match="selected camera absent"):
        inspect_muvs_train_archive(archive, expected_md5=expected_md5)


def test_rejects_archive_hash_mismatch(tmp_path):
    archive = tmp_path / "muvs.zip"
    _fixture_zip(archive)

    with pytest.raises(ValueError, match="pinned publisher MD5"):
        inspect_muvs_train_archive(archive)
