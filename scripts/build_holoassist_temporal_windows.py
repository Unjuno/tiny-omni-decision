"""Aggregate action labels into causal, fixed-duration HoloAssist windows."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from scripts.audit_holoassist_metadata import sha256_file


def build_temporal_windows(
    rows_by_split: dict[str, list[dict]], *, window_seconds: float = 30.0
) -> tuple[dict[str, list[dict]], dict[str, dict]]:
    if not math.isfinite(window_seconds) or window_seconds <= 0:
        raise ValueError("window duration must be finite and positive")
    if set(rows_by_split) != {"train", "validation"}:
        raise ValueError("train and validation rows are required")

    sessions: dict[str, dict[str, list[dict]]] = {
        "train": defaultdict(list),
        "validation": defaultdict(list),
    }
    for split, rows in rows_by_split.items():
        if not rows:
            raise ValueError(f"{split} rows cannot be empty")
        ids: set[str] = set()
        for row in rows:
            if row.get("split") != split:
                raise ValueError(f"row is incorrectly marked for {split}")
            event_id = str(row.get("id", ""))
            if not event_id or event_id in ids:
                raise ValueError(f"missing or duplicate event ID in {split}: {event_id}")
            ids.add(event_id)
            if row.get("future_context_seconds") != 0.0:
                raise ValueError("post-action context is prohibited")
            sessions[split][str(row["session_id"])].append(row)

    overlap = set(sessions["train"]) & set(sessions["validation"])
    if overlap:
        raise ValueError("train/validation sessions overlap")

    output: dict[str, list[dict]] = {"train": [], "validation": []}
    reports: dict[str, dict] = {}
    for split in ("train", "validation"):
        assigned_event_ids: set[str] = set()
        session_order = sorted(sessions[split])
        for session_id in session_order:
            events = sessions[split][session_id]
            durations = {float(row["video_duration_seconds"]) for row in events}
            task_types = {str(row["task_type"]) for row in events}
            media_hashes = {str(row["media_sha256"]) for row in events}
            media_paths = {str(row["media_path"]) for row in events}
            if (
                len(durations) != 1
                or len(task_types) != 1
                or len(media_hashes) != 1
                or len(media_paths) != 1
            ):
                raise ValueError(f"inconsistent session metadata: {session_id}")
            duration = durations.pop()
            if not math.isfinite(duration) or duration <= 0:
                raise ValueError(f"invalid video duration for {session_id}")
            task_type = task_types.pop()
            media_hash = media_hashes.pop()
            media_path = media_paths.pop()
            ordered_events = sorted(
                events,
                key=lambda row: (
                    float(row["action_end_seconds"]),
                    float(row["action_start_seconds"]),
                    str(row["id"]),
                ),
            )
            window_count = math.ceil(duration / window_seconds)
            for window_index in range(window_count):
                start = window_index * window_seconds
                end = min(duration, start + window_seconds)
                members = [
                    row
                    for row in ordered_events
                    if start < float(row["action_end_seconds"]) <= end
                ]
                member_ids = [str(row["id"]) for row in members]
                if assigned_event_ids.intersection(member_ids):
                    raise ValueError("an action event was assigned to multiple windows")
                assigned_event_ids.update(member_ids)
                incorrect_count = sum(int(row["target"]) for row in members)
                correct_count = len(members) - incorrect_count
                output[split].append(
                    {
                        "id": f"{session_id}:window-{window_index:04d}",
                        "split": split,
                        "session_id": session_id,
                        "window_index": window_index,
                        "window_count": window_count,
                        "window_start_seconds": start,
                        "window_end_seconds": end,
                        "video_duration_seconds": duration,
                        "media_path": media_path,
                        "media_sha256": media_hash,
                        "task_type": task_type,
                        "question": (
                            "Did any observed action in this time window contain an "
                            "incorrect action while performing the task ‘"
                            f"{task_type}’?"
                        ),
                        "options": ["No incorrect action", "At least one incorrect action"],
                        "target": int(incorrect_count > 0) if members else None,
                        "loss_mask": bool(members),
                        "action_count": len(members),
                        "correct_action_count": correct_count,
                        "incorrect_action_count": incorrect_count,
                        "event_ids": member_ids,
                        "future_context_seconds": 0.0,
                    }
                )
        expected_ids = {
            str(row["id"]) for events in sessions[split].values() for row in events
        }
        if assigned_event_ids != expected_ids:
            missing = sorted(expected_ids - assigned_event_ids)
            raise ValueError(f"action events fell outside time windows: {missing[:3]}")
        windows = output[split]
        labeled = [row for row in windows if row["loss_mask"]]
        reports[split] = {
            "session_count": len(session_order),
            "window_count": len(windows),
            "labeled_window_count": len(labeled),
            "unlabeled_context_window_count": len(windows) - len(labeled),
            "source_action_event_count": len(assigned_event_ids),
            "correct_action_count": sum(int(row["correct_action_count"]) for row in labeled),
            "incorrect_action_count": sum(
                int(row["incorrect_action_count"]) for row in labeled
            ),
            "positive_window_count": sum(int(row["target"] == 1) for row in labeled),
            "negative_window_count": sum(int(row["target"] == 0) for row in labeled),
            "session_ids_sha256_sorted_newline_delimited": hashlib.sha256(
                "\n".join(session_order).encode("utf-8")
            ).hexdigest(),
        }
    return output, reports


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window-seconds", type=float, default=30.0)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite temporal windows: {args.output_dir}")
    rows = {
        split: read_jsonl(args.events / f"{split}.jsonl")
        for split in ("train", "validation")
    }
    windows, report = build_temporal_windows(rows, window_seconds=args.window_seconds)
    args.output_dir.mkdir(parents=True)
    for split in ("train", "validation"):
        path = args.output_dir / f"{split}.jsonl"
        with path.open("x", encoding="utf-8", newline="\n") as output:
            for row in windows[split]:
                output.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    receipt = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_event_receipts": {
            split: sha256_file(args.events / f"{split}.jsonl")
            for split in ("train", "validation")
        },
        "window_seconds": args.window_seconds,
        "window_label_policy": (
            "Each action is assigned once to the window containing its end time; a window is "
            "positive when at least one assigned action is incorrect. Empty windows are retained "
            "as unlabelled recurrent context and excluded from loss/metrics."
        ),
        "decision_cutoff": "each window end; no future audio or video",
        "report": report,
        "train_jsonl_sha256": sha256_file(args.output_dir / "train.jsonl"),
        "validation_jsonl_sha256": sha256_file(args.output_dir / "validation.jsonl"),
    }
    (args.output_dir / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
