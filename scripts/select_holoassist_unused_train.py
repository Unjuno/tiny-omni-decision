"""Create a new session-disjoint HoloAssist dev split from unused official train sessions.

Only the official train split member is read. The former pilot's train and
validation sessions are excluded; no official validation/test IDs or labels
are read. Official train sessions are deterministically divided by task type.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import zipfile
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from scripts.audit_holoassist_metadata import sha256_file


def _read_official_train(path: Path) -> set[str]:
    with zipfile.ZipFile(path) as archive:
        matches = [
            name
            for name in archive.namelist()
            if PurePosixPath(name).name == "train-v1_2.txt"
        ]
        if len(matches) != 1:
            raise ValueError("official train split file must occur exactly once")
        return {
            value.strip()
            for value in archive.read(matches[0]).decode("utf-8").splitlines()
            if value.strip()
        }


def make_selection(
    *,
    official_train: set[str],
    annotations: list[dict[str, Any]],
    old_excluded: set[str],
    members: list[dict[str, Any]],
    seed: int,
    validation_fraction: float = 0.2,
) -> dict[str, Any]:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    by_session = {str(row["video_name"]): row for row in annotations}
    if len(by_session) != len(annotations):
        raise ValueError("duplicate trainval annotation session IDs")
    unused = official_train - old_excluded
    if not unused or not unused <= by_session.keys():
        raise ValueError("unused official-train sessions lack annotations")

    video_members = {
        str(row["session_id"]): row
        for row in members
        if str(row["name"]).endswith("/Export_py/Video_pitchshift.mp4")
    }
    if len(video_members) != sum(
        str(row["name"]).endswith("/Export_py/Video_pitchshift.mp4") for row in members
    ):
        raise ValueError("duplicate pitch-shifted video member")
    if not unused <= video_members.keys():
        raise ValueError("unused official-train session lacks indexed pitch-shifted media")

    buckets: dict[str, list[str]] = defaultdict(list)
    for session_id in sorted(unused):
        task_type = str(by_session[session_id].get("taskType", "UNKNOWN"))
        buckets[task_type].append(session_id)
    rng = random.Random(seed)
    train_ids: list[str] = []
    validation_ids: list[str] = []
    task_counts: dict[str, dict[str, int]] = {}
    for task_type, ids in sorted(buckets.items()):
        rng.shuffle(ids)
        validation_count = min(len(ids) - 1, max(1, round(len(ids) * validation_fraction)))
        validation_ids.extend(ids[:validation_count])
        train_ids.extend(ids[validation_count:])
        task_counts[task_type] = {
            "unused_official_train_sessions": len(ids),
            "train_sessions": len(ids) - validation_count,
            "development_validation_sessions": validation_count,
        }

    if set(train_ids) & set(validation_ids) or set(train_ids) | set(validation_ids) != unused:
        raise AssertionError("new split must partition all unused official train sessions")

    def describe(ids: list[str]) -> dict[str, Any]:
        videos = []
        for session_id in sorted(ids):
            member = video_members[session_id]
            videos.append(
                {
                    "session_id": session_id,
                    "task_type": str(by_session[session_id].get("taskType", "UNKNOWN")),
                    "video_member_name": str(member["name"]),
                    "offset_data": int(member["offset_data"]),
                    "size_bytes": int(member["size"]),
                }
            )
        ordered_ids = [row["session_id"] for row in videos]
        return {
            "session_count": len(videos),
            "session_id_sha256_ordered_newline_delimited": hashlib.sha256(
                "\n".join(ordered_ids).encode("utf-8")
            ).hexdigest(),
            "videos": videos,
            "video_bytes_total": sum(row["size_bytes"] for row in videos),
        }

    train = describe(train_ids)
    validation = describe(validation_ids)
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "seed": seed,
        "validation_fraction": validation_fraction,
        "split_policy": (
            "task-stratified partition of unused official-train sessions; "
            "no correctness labels used"
        ),
        "official_train_session_count": len(official_train),
        "excluded_prior_pilot_session_count": len(old_excluded),
        "unused_official_train_session_count": len(unused),
        "old_pilot_session_overlap": 0,
        "test_split_read": False,
        "validation_source": (
            "new development partition of previously unused official train sessions"
        ),
        "task_type_counts": task_counts,
        "selection": {
            "train": train,
            "validation": validation,
            "all_selected_video_bytes": train["video_bytes_total"]
            + validation["video_bytes_total"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--media-index", type=Path, required=True)
    parser.add_argument("--old-selection", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite selection: {args.output}")
    annotation = json.loads(args.annotation.read_text(encoding="utf-8"))
    index = json.loads(args.media_index.read_text(encoding="utf-8"))
    old = json.loads(args.old_selection.read_text(encoding="utf-8"))
    old_ids = {
        str(row["session_id"])
        for split in ("train",)
        for row in old["selection"][split]["videos"]
    }
    selection = make_selection(
        official_train=_read_official_train(args.splits),
        annotations=annotation,
        old_excluded=old_ids,
        members=index["index"]["members"],
        seed=args.seed,
    )
    selection["input_sha256"] = {
        "trainval_annotation": sha256_file(args.annotation),
        "official_split_archive": sha256_file(args.splits),
        "media_index": sha256_file(args.media_index),
        "prior_selection": sha256_file(args.old_selection),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
