"""Bind pitch-shifted AV media members to the immutable video pilot sessions."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from scripts.audit_holoassist_metadata import sha256_file

EXPECTED_SUFFIX = "/Export_py/Video_pitchshift.mp4"


def build_companion_plan(
    base_selection: dict, media_index: dict, allowed_session_ids: set[str]
) -> dict:
    selected = base_selection["selection"]
    members = media_index["index"]["members"]
    if media_index.get("payload_media_downloaded") is not False:
        raise ValueError("media index must be header-only; payload is fetched separately")
    companions = {}
    for member in members:
        name = str(member["name"])
        if not name.endswith(EXPECTED_SUFFIX):
            continue
        session_id = str(member["session_id"])
        if session_id in companions:
            raise ValueError(f"duplicate pitch-shifted media member: {session_id}")
        if not member.get("is_file"):
            raise ValueError(f"pitch-shifted media member is not a file: {name}")
        companions[session_id] = member

    if set(companions) != allowed_session_ids:
        raise ValueError(
            "pitch-shifted index sessions differ from the official train/validation allowlist"
        )

    expected_splits = {
        split: [str(row["session_id"]) for row in selected[split]["videos"]]
        for split in ("train", "validation")
    }
    if set(expected_splits["train"]) & set(expected_splits["validation"]):
        raise ValueError("base pilot train/validation sessions overlap")
    if len(expected_splits["train"]) != 160 or len(expected_splits["validation"]) != 207:
        raise ValueError("companion media must preserve the fixed 160/207 pilot")
    all_expected = set(expected_splits["train"]) | set(expected_splits["validation"])
    missing = all_expected - set(companions)
    if missing:
        raise ValueError(f"selected sessions lack pitch-shifted media: {sorted(missing)[:3]}")

    new_selection = {}
    for split in ("train", "validation"):
        videos = []
        for row in selected[split]["videos"]:
            session_id = str(row["session_id"])
            member = companions[session_id]
            videos.append(
                {
                    "session_id": session_id,
                    "task_type": row["task_type"],
                    "video_member_name": member["name"],
                    "offset_data": int(member["offset_data"]),
                    "size_bytes": int(member["size"]),
                }
            )
        new_selection[split] = {
            "session_count": len(videos),
            "session_ids_sha256_ordered_newline_delimited": hashlib.sha256(
                "\n".join(row["session_id"] for row in videos).encode("utf-8")
            ).hexdigest(),
            "videos": videos,
            "video_bytes_total": sum(row["size_bytes"] for row in videos),
        }
    new_selection["all_selected_video_bytes"] = sum(
        new_selection[split]["video_bytes_total"] for split in ("train", "validation")
    )
    return new_selection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-selection", type=Path, required=True)
    parser.add_argument("--media-index", type=Path, required=True)
    parser.add_argument("--allowed-sessions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite companion selection: {args.output}")
    base = json.loads(args.base_selection.read_text(encoding="utf-8"))
    index = json.loads(args.media_index.read_text(encoding="utf-8"))
    allowed = {
        line.strip()
        for line in args.allowed_sessions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if not allowed:
        raise ValueError("official train/validation session allowlist cannot be empty")
    receipt = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "base_selection_sha256": sha256_file(args.base_selection),
        "media_index_sha256": sha256_file(args.media_index),
        "allowed_session_ids_sha256_sorted_newline_delimited": hashlib.sha256(
            "\n".join(sorted(allowed)).encode("utf-8")
        ).hexdigest(),
        "selection_policy": (
            "Use exactly the fixed seed-17 train/validation session order from the base "
            "pilot and replace only each compressed-video member with its pitch-shifted "
            "companion. No test or unassigned sessions are eligible."
        ),
        "archive_url": index["archive_url"],
        "archive_size_bytes": index["archive_size_bytes"],
        "archive_etag": index["archive_etag"],
        "selection": build_companion_plan(base, index, allowed),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
