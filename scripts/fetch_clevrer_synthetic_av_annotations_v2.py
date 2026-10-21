"""Range-fetch CLEVRER official-train annotations for a frozen AV v2 split."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

from scripts.create_clevrer_synthetic_av_protocol import eligible_collision_events
from scripts.fetch_clevrer_probe import RangeFile

DATASET = "MIT-IBM/CLEVRER"
REVISION = "98b842082ba4f7c18b6b9e3f39145871782a65ef"
QUESTIONS_SHA256 = "11181da673d223f41fb596aacfbbd3ff83d39af7f09a3210549e98cb283714b4"
ANNOTATION_URL = "https://data.csail.mit.edu/clevrer/annotations/train/annotation_train.zip"
ANNOTATION_BYTES = 156_761_889
ANNOTATION_ETAG = '"957ff21-59db3309a63ea"'


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_candidate_scene_ids(preflight_path: Path) -> tuple[list[int], list[int]]:
    payload = json.loads(preflight_path.read_text(encoding="utf-8"))
    if (
        payload.get("status") != "PASS_MEDIA_AND_TASK_SPLIT_PREFLIGHT_ONLY"
        or payload.get("experiment_id") != "clevrer-synthetic-av-coverage-v2"
        or payload.get("source") != f"{DATASET}@{REVISION}"
        or payload.get("sealed_or_official_validation_test_opened") is not False
        or payload.get("candidate_validation_scenes", 0) < 200
    ):
        raise ValueError("input preflight does not establish the frozen AV v2 train source/split")
    train = payload.get("candidate_train_scene_ids")
    validation = payload.get("candidate_validation_scene_ids")
    if (
        not isinstance(train, list)
        or not isinstance(validation, list)
        or len(train) != payload.get("candidate_train_scenes")
        or len(validation) != payload.get("candidate_validation_scenes")
        or len(set(train)) != len(train)
        or len(set(validation)) != len(validation)
        or set(train) & set(validation)
    ):
        raise ValueError("input preflight scene lists are missing, duplicated, or overlapping")
    return sorted(train), sorted(validation)


def _validate_train_question_ids(path: Path, expected_ids: set[int]) -> str:
    raw = path.read_bytes()
    digest = _sha256_bytes(raw)
    if digest != QUESTIONS_SHA256:
        raise ValueError("CLEVRER train question source SHA-256 differs from the pinned revision")
    rows = json.loads(raw)
    observed = {
        row.get("scene_index")
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("scene_index"), int)
    }
    if not expected_ids <= observed:
        raise ValueError(
            "task-specific scenes are not all represented in the official train source"
        )
    return digest


def inspect_annotation_archive(
    archive_file: BinaryIO,
    train_scene_ids: list[int],
    validation_scene_ids: list[int],
) -> tuple[dict[str, bytes], dict[str, Any]]:
    all_scene_ids = set(train_scene_ids) | set(validation_scene_ids)
    if not train_scene_ids or not validation_scene_ids or set(train_scene_ids) & set(
        validation_scene_ids
    ):
        raise ValueError("nonempty, scene-disjoint train and validation IDs are required")
    with zipfile.ZipFile(archive_file) as archive:
        by_basename: dict[str, list[zipfile.ZipInfo]] = {}
        for info in archive.infolist():
            if info.filename.casefold().endswith(".json"):
                basename = PurePosixPath(info.filename).name.casefold()
                by_basename.setdefault(basename, []).append(info)
        payloads: dict[str, bytes] = {}
        event_counts: dict[str, int] = {}
        eligible_scene_counts = {"train": 0, "validation": 0}
        per_split_events = {"train": 0, "validation": 0}
        scene_to_split = {
            **{scene: "train" for scene in train_scene_ids},
            **{scene: "validation" for scene in validation_scene_ids},
        }
        for scene in sorted(all_scene_ids):
            basename = f"annotation_{scene:05d}.json".casefold()
            matches = by_basename.get(basename, [])
            if len(matches) != 1:
                raise ValueError(
                    "expected exactly one official annotation for scene "
                    f"{scene}, found {len(matches)}"
                )
            info = matches[0]
            data = archive.read(info)  # zipfile verifies the member CRC.
            annotation = json.loads(data)
            if (
                annotation.get("scene_index") != scene
                or annotation.get("video_filename") != f"video_{scene:05d}.mp4"
            ):
                raise ValueError(f"annotation identity mismatch for official train scene {scene}")
            events = eligible_collision_events(annotation)
            split = scene_to_split[scene]
            event_counts[str(scene)] = len(events)
            if events:
                eligible_scene_counts[split] += 1
                per_split_events[split] += len(events)
            payloads[f"annotation_{scene:05d}.json"] = data
    return payloads, {
        "scene_count": len(all_scene_ids),
        "train_scene_count": len(train_scene_ids),
        "validation_scene_count": len(validation_scene_ids),
        "eligible_scene_counts": eligible_scene_counts,
        "eligible_collision_counts": per_split_events,
        "eligible_event_count_by_scene": event_counts,
        "archive_member_crc_checks": len(payloads),
    }


def fetch_annotations(
    *,
    preflight_path: Path,
    train_questions_path: Path,
    output_dir: Path,
    archive_file: BinaryIO | None = None,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse or overwrite annotation output: {output_dir}")
    train_scene_ids, validation_scene_ids = _read_candidate_scene_ids(preflight_path)
    questions_sha = _validate_train_question_ids(
        train_questions_path, set(train_scene_ids) | set(validation_scene_ids)
    )
    owned_archive = archive_file is None
    if owned_archive:
        archive_file = RangeFile(ANNOTATION_URL, ANNOTATION_BYTES, ANNOTATION_ETAG)
    try:
        payloads, counts = inspect_annotation_archive(
            archive_file, train_scene_ids, validation_scene_ids
        )
        if counts["eligible_scene_counts"]["validation"] < 200:
            raise ValueError(
                "frozen validation split has fewer than 200 eligible video groups; "
                "do not alter the split after inspecting annotations"
            )
        output_dir.mkdir(parents=True)
        annotation_dir = output_dir / "annotations"
        annotation_dir.mkdir()
        entries = []
        for name, payload in sorted(payloads.items()):
            path = annotation_dir / name
            with path.open("xb") as stream:
                stream.write(payload)
            entries.append(
                {
                    "file": f"annotations/{name}",
                    "bytes": len(payload),
                    "sha256": _sha256_bytes(payload),
                }
            )
        report = {
            "status": "PASS_PINNED_TRAIN_ANNOTATIONS_EXTRACTED",
            "experiment_id": "clevrer-synthetic-av-coverage-v2",
            "source": f"{DATASET}@{REVISION}",
            "license": "CC0-1.0",
            "official_split": "train",
            "train_question_source_sha256": questions_sha,
            "preflight_sha256": _sha256_bytes(preflight_path.read_bytes()),
            "annotation_archive": {
                "url": ANNOTATION_URL,
                "size_bytes": ANNOTATION_BYTES,
                "etag": ANNOTATION_ETAG,
                "full_archive_sha256": "NOT_COMPUTED_RANGE_READ_ONLY",
                "range_bytes_read": getattr(archive_file, "bytes_read", None),
            },
            "sealed_or_official_validation_test_opened": False,
            "validation_is_task_specific_not_media_unseen_or_blind": True,
            **counts,
            "annotation_payload_bytes": sum(item["bytes"] for item in entries),
            "entries": entries,
        }
        receipt_path = output_dir / "receipt.json"
        with receipt_path.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(report, stream, indent=2, sort_keys=True)
            stream.write("\n")
        return report
    finally:
        if owned_archive:
            session = getattr(archive_file, "session", None)
            if session is not None:
                session.close()
            archive_file.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--train-questions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = fetch_annotations(
        preflight_path=args.preflight,
        train_questions_path=args.train_questions,
        output_dir=args.output_dir,
    )
    summary = {
        key: value
        for key, value in report.items()
        if key not in {"entries", "eligible_event_count_by_scene"}
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
