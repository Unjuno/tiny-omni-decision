"""Train-only metadata and label-integrity audit for the MUVS Zenodo deposit."""

from __future__ import annotations

import csv
import hashlib
import io
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

MUVS_DATA_ZIP_MD5 = "8fd1d426e0b4308655ea71884d1b0c8e"
TRAIN_PREFIX = "data/fragments/train/"
TEST_PREFIX = "data/fragments/test/"
ANNOTATION_PREFIX = "data/annotations/"
FRAGMENT_SUFFIX = "_FRAGMENTS.csv"
ANNOTATION_SUFFIX = "_ANNOTATIONS.csv"
REQUIRED_FRAGMENT_COLUMNS = {
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
}
REQUIRED_ANNOTATION_COLUMNS = {
    "dataset_type",
    "event",
    "period",
    "fragment_number",
    "camera_selected",
    "offsets",
}


def _event_from_filename(name: str, *, prefix: str, suffix: str) -> str:
    if not name.startswith(prefix) or not name.endswith(suffix):
        raise ValueError(f"unexpected MUVS member path: {name}")
    event = name[len(prefix) : -len(suffix)]
    if not event or "/" in event or "\\" in event:
        raise ValueError(f"invalid event identifier in {name}")
    return event


def _read_csv_member(archive: zipfile.ZipFile, name: str) -> list[dict[str, str]]:
    with archive.open(name, "r") as stream:
        text = io.TextIOWrapper(stream, encoding="utf-8-sig", newline="")
        reader = csv.DictReader(text)
        if reader.fieldnames is None:
            raise ValueError(f"CSV header missing: {name}")
        return list(reader)


def _target_prior_baseline(
    event_rows: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    def summarize(*, require_multiple_choices: bool) -> dict[str, Any]:
        selected = {
            event: [
                row
                for row in rows
                if not require_multiple_choices or len(row["choices"]) >= 2
            ]
            for event, rows in event_rows.items()
        }
        pooled = Counter(row["target"] for rows in selected.values() for row in rows)
        correct = 0
        total = 0
        event_accuracy: dict[str, float] = {}
        for held_event, rows in sorted(selected.items()):
            training_counts = Counter(
                row["target"]
                for event, other_rows in selected.items()
                if event != held_event
                for row in other_rows
            )
            event_correct = 0
            for row in rows:
                prediction = min(
                    row["choices"],
                    key=lambda option: (-training_counts[option], option),
                )
                event_correct += int(prediction == row["target"])
            event_total = len(rows)
            correct += event_correct
            total += event_total
            event_accuracy[held_event] = (
                event_correct / event_total if event_total else 0.0
            )

        chance_sum = sum(
            1 / len(row["choices"]) for rows in selected.values() for row in rows
        )
        return {
            "pooled_target_counts": dict(sorted(pooled.items())),
            "leave_one_event_out_global_camera_prior": {
                "accuracy": correct / total if total else 0.0,
                "correct": correct,
                "count": total,
                "event_accuracy": event_accuracy,
            },
            "uniform_choice_expected_accuracy": chance_sum / total if total else 0.0,
        }

    return {
        "all_labeled_rows": summarize(require_multiple_choices=False),
        "at_least_two_available_options": summarize(require_multiple_choices=True),
    }


def inspect_muvs_train_archive(
    archive_path: str | Path,
    *,
    expected_md5: str = MUVS_DATA_ZIP_MD5,
) -> dict[str, Any]:
    """Read only publisher-train fragment and matching annotation CSV payloads.

    Test ZIP members are identified from central-directory names for split
    accounting, but their bytes are never opened or parsed.
    """
    archive_path = Path(archive_path)
    digest = hashlib.md5(usedforsecurity=False)
    sha256 = hashlib.sha256()
    with archive_path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
            sha256.update(chunk)
    md5 = digest.hexdigest()
    if md5 != expected_md5:
        raise ValueError("MUVS data.zip does not match the pinned publisher MD5")

    with zipfile.ZipFile(archive_path) as archive:
        member_names = [item.filename for item in archive.infolist()]
        train_members = sorted(
            name
            for name in member_names
            if name.startswith(TRAIN_PREFIX) and name.endswith(FRAGMENT_SUFFIX)
        )
        test_members = sorted(
            name
            for name in member_names
            if name.startswith(TEST_PREFIX) and name.endswith(FRAGMENT_SUFFIX)
        )
        annotation_members = {
            _event_from_filename(
                name, prefix=ANNOTATION_PREFIX, suffix=ANNOTATION_SUFFIX
            ): name
            for name in member_names
            if name.startswith(ANNOTATION_PREFIX) and name.endswith(ANNOTATION_SUFFIX)
        }
        train_events = {
            _event_from_filename(name, prefix=TRAIN_PREFIX, suffix=FRAGMENT_SUFFIX)
            for name in train_members
        }
        test_events = {
            _event_from_filename(name, prefix=TEST_PREFIX, suffix=FRAGMENT_SUFFIX)
            for name in test_members
        }
        if not train_members or not test_members:
            raise ValueError("publisher train and test fragment inventories are required")
        if train_events & test_events:
            raise ValueError("an event appears in both publisher partitions")

        event_rows: dict[str, list[dict[str, Any]]] = {}
        event_counts: dict[str, dict[str, Any]] = {}
        missing_sensor_rows = Counter()
        active_camera_mismatches = []
        sensor_columns = sorted(
            REQUIRED_FRAGMENT_COLUMNS
            & {
                "Accelerometer_x",
                "Accelerometer_y",
                "Accelerometer_z",
                "Gyroscope_x",
                "Gyroscope_y",
                "Gyroscope_z",
            }
        )
        total_fragment_rows = 0
        total_fragment_payload_bytes = 0
        total_annotation_payload_bytes = 0

        for member in train_members:
            event = _event_from_filename(
                member, prefix=TRAIN_PREFIX, suffix=FRAGMENT_SUFFIX
            )
            annotation_member = annotation_members.get(event)
            if annotation_member is None:
                raise ValueError(f"matching annotation CSV missing for train event {event}")

            fragment_info = archive.getinfo(member)
            total_fragment_payload_bytes += fragment_info.file_size
            fragment_rows = _read_csv_member(archive, member)
            if not fragment_rows:
                raise ValueError(f"empty train fragment CSV: {member}")
            if not REQUIRED_FRAGMENT_COLUMNS.issubset(fragment_rows[0]):
                raise ValueError(f"required fragment columns missing: {member}")
            total_fragment_rows += len(fragment_rows)

            candidates: dict[tuple[str, str], set[str]] = {}
            active_counts: dict[tuple[str, str], set[int]] = {}
            for row in fragment_rows:
                if row["event_id"] != event:
                    raise ValueError(f"event identity mismatch in {member}")
                fragment_number, separator, _camera_source = row["fragment_id"].partition(
                    "_cam_"
                )
                if not separator or not fragment_number.isdigit():
                    raise ValueError(f"invalid fragment ID in {member}")
                key = (row["period"], fragment_number)
                candidates.setdefault(key, set()).add(row["camera_id"])
                try:
                    active_counts.setdefault(key, set()).add(int(row["ActiveCameras"]))
                except ValueError as exc:
                    raise ValueError(f"invalid ActiveCameras value in {member}") from exc
                for column in sensor_columns:
                    if not row[column].strip():
                        missing_sensor_rows[column] += 1

            total_annotation_payload_bytes += archive.getinfo(annotation_member).file_size
            annotations = _read_csv_member(archive, annotation_member)
            if not annotations:
                raise ValueError(f"empty train annotation CSV: {annotation_member}")
            if not REQUIRED_ANNOTATION_COLUMNS.issubset(annotations[0]):
                raise ValueError(f"required annotation columns missing: {annotation_member}")

            instances: list[dict[str, Any]] = []
            seen_keys: set[tuple[str, str]] = set()
            for label in annotations:
                if label["dataset_type"] != "train":
                    raise ValueError(f"non-train annotation row in {annotation_member}")
                if label["event"] != f"{event}_filtered":
                    raise ValueError(f"annotation event identity mismatch: {annotation_member}")
                key = (label["period"], label["fragment_number"])
                if key in seen_keys:
                    raise ValueError(f"duplicate train decision identity in {annotation_member}")
                seen_keys.add(key)
                choices = candidates.get(key)
                if not choices:
                    raise ValueError(
                        f"annotation has no matching train camera choices: {event} {key}"
                    )
                if label["camera_selected"] not in choices:
                    raise ValueError(
                        f"selected camera absent from train candidate set: {event} {key}"
                    )
                expected_active = active_counts.get(key, set())
                if expected_active != {len(choices)}:
                    active_camera_mismatches.append(
                        {
                            "event": event,
                            "period": key[0],
                            "fragment_number": key[1],
                            "unique_camera_ids": len(choices),
                            "publisher_active_camera_values": sorted(expected_active),
                        }
                    )
                instances.append(
                    {
                        "period": key[0],
                        "fragment_number": key[1],
                        "target": label["camera_selected"],
                        "choices": sorted(choices),
                        "choice_count": len(choices),
                    }
                )
            event_rows[event] = instances
            event_counts[event] = {
                "fragment_rows": len(fragment_rows),
                "decision_count": len(instances),
                "fragment_group_count": len(candidates),
                "unlabeled_fragment_group_count": len(set(candidates) - seen_keys),
                "camera_choice_counts": dict(
                    sorted(Counter(row["choice_count"] for row in instances).items())
                ),
                "selected_camera_counts": dict(
                    sorted(Counter(row["target"] for row in instances).items())
                ),
            }

    decisions = [row for rows in event_rows.values() for row in rows]
    return {
        "schema": "muvs_train_metadata_preflight_v1",
        "source": {
            "record_id": "20708683",
            "version": "v1.0",
            "deposit_license": "CC BY 4.0",
            "archive_bytes": archive_path.stat().st_size,
            "archive_md5": md5,
            "archive_sha256": sha256.hexdigest(),
            "train_event_count_from_fragment_files": len(train_events),
            "test_event_count_from_fragment_filenames_only": len(test_events),
            "train_test_event_overlap": sorted(train_events & test_events),
            "train_events": sorted(train_events),
            "test_event_names_listed_for_split_audit": True,
            "test_fragment_member_contents_decompressed": False,
            "test_annotation_member_contents_decompressed": False,
            "test_partition_labels_or_sensor_values_parsed": False,
            "external_gcs_video_accessed": False,
        },
        "task": {
            "question": "Which camera is the best view for this fragment?",
            "options": "camera IDs present for the matching event/period/fragment number",
            "target": "camera_selected",
            "train_event_count": len(event_rows),
            "train_decision_count": len(decisions),
            "choice_count_distribution": dict(
                sorted(Counter(row["choice_count"] for row in decisions).items())
            ),
            "single_option_rows_not_decision_eligible": sum(
                row["choice_count"] < 2 for row in decisions
            ),
            "multi_option_decision_count": sum(
                row["choice_count"] >= 2 for row in decisions
            ),
            "selected_camera_counts": dict(
                sorted(Counter(row["target"] for row in decisions).items())
            ),
            "event_counts": event_counts,
            "train_fragment_rows_read": total_fragment_rows,
            "train_fragment_payload_bytes_read": total_fragment_payload_bytes,
            "train_annotation_payload_bytes_read": total_annotation_payload_bytes,
            "sensor_missing_cells_by_axis": dict(sorted(missing_sensor_rows.items())),
            "sensor_rows_missing_fraction_by_axis": {
                name: count / total_fragment_rows if total_fragment_rows else 0.0
                for name, count in sorted(missing_sensor_rows.items())
            },
            "active_camera_count_mismatches": active_camera_mismatches,
            "train_only_metadata_prior_baseline": _target_prior_baseline(event_rows),
            "labels_verified_against_candidate_sets": True,
            "annotator_count_or_agreement": "not reported in the inspected metadata",
            "video_quality_usefulness": "not assessed; GCS video license scope unresolved",
        },
        "publisher_readme_split_event_claim": {"train": 11, "test": 6},
        "split_count_discrepancy": {
            "status": len(train_events) != 11 or len(test_events) != 6,
            "archive_train_fragment_events": len(train_events),
            "archive_test_fragment_events": len(test_events),
        },
    }

