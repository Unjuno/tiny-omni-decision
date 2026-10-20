"""Audit CaptainCook4D train/validation metadata without test-set metrics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def audit_source(annotation_root: Path, downloader_metadata: Path) -> dict[str, Any]:
    split_path = annotation_root / "data_splits" / "person_data_split_combined.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    # The test key is deliberately never accessed.
    train_ids = set(map(str, split["train"]))
    validation_ids = set(map(str, split["val"]))
    if not train_ids or not validation_ids or train_ids & validation_ids:
        raise ValueError("person split must have nonempty, disjoint train/val IDs")

    metadata_path = annotation_root / "metadata" / "video_information.csv"
    metadata_rows = read_csv(metadata_path)
    metadata = {row["recording_id"]: row for row in metadata_rows}
    if len(metadata) != len(metadata_rows):
        raise ValueError("duplicate recording IDs in video metadata")
    selected_ids = train_ids | validation_ids
    if not selected_ids <= metadata.keys():
        raise ValueError("a train/validation recording lacks metadata")

    activity_path = annotation_root / "annotation_csv" / "recording_id_step_idx.csv"
    activity_rows = read_csv(activity_path)
    activity = {row["recording_id"]: row["activity_id"] for row in activity_rows}
    if not selected_ids <= activity.keys():
        raise ValueError("a train/validation recording lacks activity mapping")

    train_people = {metadata[rid]["person_id"] for rid in train_ids}
    val_people = {metadata[rid]["person_id"] for rid in validation_ids}
    shared_people = train_people & val_people
    shared_train_ids = sorted(
        rid for rid in train_ids if metadata[rid]["person_id"] in shared_people
    )
    repaired_train = train_ids - set(shared_train_ids)

    step_path = annotation_root / "annotation_csv" / "step_annotations.csv"
    counts = {name: Counter() for name in ("train", "validation")}
    recording_counts = {name: set() for name in ("train", "validation")}
    with step_path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            recording_id = row["recording_id"]
            if recording_id not in selected_ids:
                continue
            split_name = "train" if recording_id in train_ids else "validation"
            label = row["has_errors"].strip().casefold()
            if label not in {"true", "false", "1", "0"}:
                raise ValueError(f"invalid has_errors label in {recording_id}")
            counts[split_name]["positive" if label in {"true", "1"} else "negative"] += 1
            recording_counts[split_name].add(recording_id)

    links = json.loads(downloader_metadata.read_text(encoding="utf-8"))
    coverage = {}
    for name, ids in (("train", train_ids), ("validation", validation_ids)):
        coverage[name] = {
            "recordings": len(ids),
            "recordings_with_labeled_steps": len(recording_counts[name]),
            "labeled_steps": sum(counts[name].values()),
            "positive_steps": counts[name]["positive"],
            "negative_steps": counts[name]["negative"],
            "distinct_activities": len({activity[rid] for rid in ids}),
            "gopro_360p_available": sum(bool(links[rid].get("gopro_360p")) for rid in ids),
            "hololens_sync_pv_available": sum(
                bool(links[rid].get("hololens_sync_pv_video")) for rid in ids
            ),
        }
    train_activities = {activity[rid] for rid in train_ids}
    val_activities = {activity[rid] for rid in validation_ids}
    train_env = {metadata[rid]["environment_id"] for rid in train_ids}
    val_env = {metadata[rid]["environment_id"] for rid in validation_ids}
    repaired_people = {metadata[rid]["person_id"] for rid in repaired_train}

    return {
        "schema_version": 1,
        "official_annotations_revision": "a8a920a3293c4db27099a20ddbe3a3a9be1283e3",
        "official_downloader_revision": "cf248fe2139fc6e37f9737141ec719948fbd1747",
        "official_test_split_metrics_loaded": False,
        "media_downloaded": False,
        "input_sha256": {
            "person_split": sha256_file(split_path),
            "video_metadata": sha256_file(metadata_path),
            "activity_mapping": sha256_file(activity_path),
            "step_annotations": sha256_file(step_path),
            "downloader_metadata": sha256_file(downloader_metadata),
        },
        "split_audit": {
            "train_recordings": len(train_ids),
            "validation_recordings": len(validation_ids),
            "recording_overlap": len(train_ids & validation_ids),
            "person_overlap": len(shared_people),
            "person_overlap_ids": sorted(shared_people),
            "shared_person_train_recordings": shared_train_ids,
            "train_after_removing_shared_person_recordings": len(repaired_train),
            "person_overlap_after_removal": len(repaired_people & val_people),
            "environment_overlap": len(train_env & val_env),
            "activity_overlap": len(train_activities & val_activities),
            "train_activity_count": len(train_activities),
            "validation_activity_count": len(val_activities),
            "train_ids_sha256_sorted_newline": hashlib.sha256(
                "\n".join(sorted(train_ids)).encode()
            ).hexdigest(),
            "validation_ids_sha256_sorted_newline": hashlib.sha256(
                "\n".join(sorted(validation_ids)).encode()
            ).hexdigest(),
        },
        "coverage": coverage,
        "task_disposition": {
            "video_text_step_conformance": "PROVISIONAL_SCREEN_ONLY",
            "audio_video_complementarity": "NOT_ESTABLISHED",
            "reason": (
                "step-error labels are not marked as requiring audio; "
                "labels do not establish complementary audiovisual necessity"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--downloader-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite audit: {args.output}")
    report = audit_source(args.annotations, args.downloader_metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
