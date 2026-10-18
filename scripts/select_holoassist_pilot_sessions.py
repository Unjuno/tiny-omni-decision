"""Select a deterministic, task-type-stratified HoloAssist session pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import zipfile
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

from scripts.audit_holoassist_metadata import SPLIT_FILES, sha256_file, stable_id_hash


def load_splits(path: Path) -> dict[str, set[str]]:
    with zipfile.ZipFile(path) as archive:
        names = {Path(member).name: member for member in archive.namelist()}
        result = {}
        for split, filename in SPLIT_FILES.items():
            if filename not in names:
                raise ValueError(f"missing official {split} split file")
            result[split] = {
                value.strip()
                for value in archive.read(names[filename]).decode("utf-8").splitlines()
                if value.strip()
            }
    if sum(map(len, result.values())) != len(set.union(*result.values())):
        raise ValueError("official split session IDs overlap")
    return result


def _stratified_sample(
    ids: set[str], records: dict[str, dict], count: int, seed: int
) -> list[str]:
    if count < 1 or count > len(ids):
        raise ValueError(f"requested {count} sessions from a split with {len(ids)} IDs")
    buckets: dict[str, list[str]] = defaultdict(list)
    for session_id in sorted(ids):
        buckets[str(records[session_id].get("taskType", "UNKNOWN"))].append(session_id)
    rng = random.Random(seed)
    for values in buckets.values():
        rng.shuffle(values)
    selected: list[str] = []
    task_types = sorted(buckets)
    while len(selected) < count:
        changed = False
        for task_type in task_types:
            if buckets[task_type] and len(selected) < count:
                selected.append(buckets[task_type].pop())
                changed = True
        if not changed:
            break
    return selected


def build_selection(
    records: list[dict],
    splits: dict[str, set[str]],
    video_members: list[dict],
    *,
    train_count: int,
    validation_count: int,
    seed: int = 17,
) -> dict:
    by_id = {str(row["video_name"]): row for row in records}
    if len(by_id) != len(records):
        raise ValueError("duplicate annotation session IDs")
    if set(by_id) & splits["test"]:
        raise ValueError("test annotation rows must not be present in trainval annotations")
    if not splits["train"] | splits["validation"] <= by_id.keys():
        raise ValueError("train/validation session lacks annotation")
    member_by_session = {}
    for member in video_members:
        if not str(member["name"]).lower().endswith("/video_compress.mp4"):
            continue
        session_id = str(member["session_id"])
        if session_id in member_by_session:
            raise ValueError(f"duplicate compressed video for {session_id}")
        if session_id in splits["test"]:
            raise ValueError("test media path must not be present in the supplied media index")
        member_by_session[session_id] = member
    allowed = splits["train"] | splits["validation"]
    if set(member_by_session) != allowed:
        raise ValueError("train/validation sessions and indexed compressed videos differ")

    train_ids = _stratified_sample(splits["train"], by_id, train_count, seed)
    validation_ids = _stratified_sample(
        splits["validation"], by_id, validation_count, seed + 1
    )
    if set(train_ids) & set(validation_ids):
        raise ValueError("selected train and validation sessions overlap")

    def describe(selected_ids: list[str]) -> dict:
        task_counts = Counter(str(by_id[sid].get("taskType", "UNKNOWN")) for sid in selected_ids)
        correctness = Counter()
        videos = []
        for session_id in selected_ids:
            for event in by_id[session_id].get("events", []):
                if event.get("label") == "Fine grained action":
                    label = str(event.get("attributes", {}).get("Action Correctness", "UNKNOWN"))
                    correctness[label] += 1
            member = member_by_session[session_id]
            videos.append(
                {
                    "session_id": session_id,
                    "task_type": str(by_id[session_id].get("taskType", "UNKNOWN")),
                    "video_member_name": member["name"],
                    "offset_data": int(member["offset_data"]),
                    "size_bytes": int(member["size"]),
                }
            )
        return {
            "session_count": len(selected_ids),
            "session_id_sha256_ordered_newline_delimited": hashlib.sha256(
                "\n".join(selected_ids).encode("utf-8")
            ).hexdigest(),
            "session_id_sha256_sorted_newline_delimited": stable_id_hash(set(selected_ids)),
            "task_type_counts": dict(sorted(task_counts.items())),
            "fine_action_label_counts": dict(sorted(correctness.items())),
            "video_bytes_total": sum(row["size_bytes"] for row in videos),
            "videos": videos,
        }

    return {
        "seed": seed,
        "train": describe(train_ids),
        "validation": describe(validation_ids),
        "all_selected_video_bytes": sum(
            row["size_bytes"]
            for split in ("train", "validation")
            for row in describe(train_ids if split == "train" else validation_ids)["videos"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--media-index", type=Path, required=True)
    parser.add_argument("--train-sessions", type=int, default=320)
    parser.add_argument("--validation-sessions", type=int, default=207)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite selection plan: {args.output}")
    records = json.loads(args.annotation.read_text(encoding="utf-8"))
    index = json.loads(args.media_index.read_text(encoding="utf-8"))
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "input_hashes": {
            "annotation_sha256": sha256_file(args.annotation),
            "split_zip_sha256": sha256_file(args.splits),
            "media_index_sha256": sha256_file(args.media_index),
        },
        "selection_policy": {
            "seed": args.seed,
            "train_selection": (
                "round-robin task-type stratification over official train sessions"
            ),
            "validation_selection": (
                "round-robin task-type stratification over official validation sessions"
            ),
            "test_sessions_or_media_included": False,
            "validation_target_labels_used_for_selection": False,
        },
        "selection": build_selection(
            records,
            load_splits(args.splits),
            index["index"]["members"],
            train_count=args.train_sessions,
            validation_count=args.validation_sessions,
            seed=args.seed,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
