"""Audit HoloAssist split and annotation metadata without opening media or test labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

SPLIT_FILES = {
    "train": "train-v1_2.txt",
    "validation": "val-v1_2.txt",
    "test": "test-v1_2.txt",
}
ACTION_LABEL = "Fine grained action"
CORRECTNESS_KEY = "Action Correctness"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_id_hash(ids: set[str]) -> str:
    payload = "\n".join(sorted(ids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def duration_seconds(raw: object) -> float | None:
    if not isinstance(raw, str):
        return None
    match = re.fullmatch(r"(\d+):(\d{2}):(\d{2})(?:\.(\d+))?", raw)
    if match is None:
        return None
    hours, minutes, seconds, fraction = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + float(
        f"0.{fraction or '0'}"
    )


def load_splits(path: Path) -> dict[str, set[str]]:
    with zipfile.ZipFile(path) as archive:
        members = {Path(name).name: name for name in archive.namelist()}
        splits: dict[str, set[str]] = {}
        for split, filename in SPLIT_FILES.items():
            if filename not in members:
                raise ValueError(f"missing official {split} split file: {filename}")
            values = {
                line.strip()
                for line in archive.read(members[filename]).decode("utf-8").splitlines()
                if line.strip()
            }
            if not values:
                raise ValueError(f"official {split} split is empty")
            splits[split] = values
    if sum(map(len, splits.values())) != len(set.union(*splits.values())):
        raise ValueError("official split session IDs overlap")
    return splits


def audit_metadata(records: list[dict], splits: dict[str, set[str]]) -> dict:
    if set(splits) != set(SPLIT_FILES):
        raise ValueError("train, validation, and test split IDs are required")
    split_ids = list(splits.values())
    if sum(map(len, split_ids)) != len(set.union(*split_ids)):
        raise ValueError("official split session IDs overlap")
    session_ids = [str(record["video_name"]) for record in records]
    task_ids = [str(record["taskId"]) for record in records]
    if len(session_ids) != len(set(session_ids)):
        raise ValueError("annotation records contain duplicate video_name IDs")
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("annotation records contain duplicate taskId values")

    annotation_ids = set(session_ids)
    split_union = set.union(*splits.values())
    outside_annotations = annotation_ids - split_union
    missing_train_validation = (splits["train"] | splits["validation"]) - annotation_ids
    if missing_train_validation:
        raise ValueError("train/validation split IDs lack trainval annotations")

    by_session = {str(record["video_name"]): record for record in records}
    summary: dict[str, object] = {
        "annotation_record_count": len(records),
        "unique_annotation_session_count": len(annotation_ids),
        "unique_task_id_count": len(set(task_ids)),
        "unassigned_annotation_count": len(outside_annotations),
        "annotation_sessions_missing_from_train_or_validation": len(missing_train_validation),
        "split_overlap_counts": {
            f"{left}_x_{right}": len(splits[left] & splits[right])
            for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
        },
        "splits": {},
    }
    split_results: dict[str, dict[str, object]] = {}
    for split in ("train", "validation", "test"):
        ids = splits[split]
        result: dict[str, object] = {
            "session_count": len(ids),
            "session_id_sha256_sorted_newline_delimited": stable_id_hash(ids),
            "annotation_coverage_count": len(ids & annotation_ids),
        }
        if split != "test":
            actions: Counter[str] = Counter()
            labels: Counter[str] = Counter()
            task_types: Counter[str] = Counter()
            reversed_intervals = 0
            out_of_range_intervals = 0
            action_interval_count = 0
            for session_id in sorted(ids):
                record = by_session[session_id]
                task_types[str(record.get("taskType", "UNKNOWN"))] += 1
                duration = duration_seconds(
                    record.get("videoMetadata", {}).get("duration", {}).get("raw")
                )
                if duration is None:
                    raise ValueError(f"unparseable raw duration for {session_id}")
                for event in record.get("events", []):
                    label = event.get("label")
                    labels[str(label)] += 1
                    if label != ACTION_LABEL:
                        continue
                    action_interval_count += 1
                    correctness = str(event.get("attributes", {}).get(CORRECTNESS_KEY, "UNKNOWN"))
                    actions[correctness] += 1
                    start, end = event.get("start"), event.get("end")
                    if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
                        out_of_range_intervals += 1
                    elif end < start:
                        reversed_intervals += 1
                    elif start < 0 or end - duration > 0.05:
                        out_of_range_intervals += 1
            result.update(
                {
                    "task_type_session_counts": dict(sorted(task_types.items())),
                    "event_label_counts": dict(sorted(labels.items())),
                    "fine_grained_action_count": action_interval_count,
                    "fine_grained_action_correctness_counts": dict(sorted(actions.items())),
                    "reversed_fine_action_intervals": reversed_intervals,
                    "out_of_range_fine_action_intervals": out_of_range_intervals,
                }
            )
        split_results[split] = result
    summary["splits"] = split_results
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite audit receipt: {args.output}")
    records = json.loads(args.annotation.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not all(isinstance(row, dict) for row in records):
        raise ValueError("expected a list of train/validation annotation records")
    splits = load_splits(args.splits)
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "input_hashes": {
            "trainval_annotation_sha256": sha256_file(args.annotation),
            "official_splits_zip_sha256": sha256_file(args.splits),
        },
        "policy": {
            "test_split_handling": (
                "session IDs used only for split-disjointness; no test labels or media read"
            ),
            "media_opened_or_decoded": False,
            "participant_identity_available": False,
            "participant_disjointness_verified": False,
        },
        "audit": audit_metadata(records, splits),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
