"""Build a video/audio/text action-correctness Decision corpus from pilot media."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import zipfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from scripts.audit_holoassist_metadata import SPLIT_FILES, duration_seconds, sha256_file

OPTION_TEXT = ["The action is correct.", "The action is incorrect."]
WRONG_LABELS = {
    "Wrong Action, corrected by instructor verbally",
    "Wrong Action, corrected by student",
    "Wrong Action, not corrected",
}


def load_official_splits(path: Path) -> dict[str, set[str]]:
    with zipfile.ZipFile(path) as archive:
        members = {Path(name).name: name for name in archive.namelist()}
        result = {}
        for split, name in SPLIT_FILES.items():
            if name not in members:
                raise ValueError(f"missing official {split} split list")
            result[split] = {
                line.strip()
                for line in archive.read(members[name]).decode("utf-8").splitlines()
                if line.strip()
            }
    if sum(map(len, result.values())) != len(set.union(*result.values())):
        raise ValueError("official split session IDs overlap")
    return result


def load_official_train_only(path: Path) -> set[str]:
    """Read only the official train session list for an internal dev split."""
    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist() if Path(name).name == SPLIT_FILES["train"]]
        if len(members) != 1:
            raise ValueError("official train split list must occur exactly once")
        return {
            line.strip()
            for line in archive.read(members[0]).decode("utf-8").splitlines()
            if line.strip()
        }


def validate_av_validation_report(
    report: dict,
    *,
    selection_sha256: str,
    media_manifest_sha256: str,
    expected_media: set[tuple[str, str, str]],
) -> None:
    """Require a successful AV probe for every selected media file."""
    if report.get("status") != "PASS_HOLOASSIST_TRAINVAL_AV_MEDIA":
        raise ValueError("AV media validation report is missing or not passing")
    if report.get("selection_sha256") != selection_sha256:
        raise ValueError("AV validation report uses a different selection plan")
    if report.get("media_manifest_sha256") != media_manifest_sha256:
        raise ValueError("AV validation report uses a different downloaded-media manifest")
    rows = report.get("media")
    if not isinstance(rows, list):
        raise ValueError("AV validation report has no per-file results")
    validated = {
        (
            str(row.get("split")),
            str(row.get("session_id")),
            str(row.get("media_sha256")),
        )
        for row in rows
        if isinstance(row, dict)
    }
    if len(rows) != len(expected_media) or len(validated) != len(rows):
        raise ValueError("AV validation report has missing or duplicate media rows")
    if validated != expected_media:
        raise ValueError("AV validation report does not exactly cover selected media hashes")
    if any(
        not isinstance(row, dict)
        or not isinstance(row.get("video_duration_seconds"), (int, float))
        or not math.isfinite(float(row["video_duration_seconds"]))
        or float(row["video_duration_seconds"]) <= 0
        for row in rows
    ):
        raise ValueError("AV validation report has missing or invalid video durations")


def video_durations_from_av_report(report: dict) -> dict[tuple[str, str], float]:
    """Read exact media-end timestamps from a validated per-file AV report."""
    durations = {
        (str(row["split"]), str(row["session_id"])): float(row["video_duration_seconds"])
        for row in report["media"]
    }
    if len(durations) != len(report["media"]):
        raise ValueError("AV validation report contains duplicate video durations")
    return durations


def build_event_rows(
    *,
    records: list[dict],
    split_ids: dict[str, set[str]],
    selected_sessions: dict[str, list[dict]],
    media_files: dict[tuple[str, str], dict],
    media_root: Path,
    media_video_durations: dict[tuple[str, str], float] | None = None,
    pre_context_seconds: float = 0.5,
) -> tuple[dict[str, list[dict]], dict[str, dict]]:
    if pre_context_seconds < 0:
        raise ValueError("pre-action context must not be negative")
    by_session = {str(row["video_name"]): row for row in records}
    if len(by_session) != len(records):
        raise ValueError("duplicate annotation session ID")
    if set(by_session) & split_ids["test"]:
        raise ValueError("test annotation records are prohibited")

    output: dict[str, list[dict]] = {"train": [], "validation": []}
    report: dict[str, dict] = {}
    for split in ("train", "validation"):
        session_key = "validation" if split == "validation" else "train"
        rows: list[dict] = []
        reasons: Counter[str] = Counter()
        label_counts: Counter[str] = Counter()
        seen_ids: set[str] = set()
        selected_ids = [str(item["session_id"]) for item in selected_sessions[session_key]]
        if len(selected_ids) != len(set(selected_ids)):
            raise ValueError(f"duplicate selected session in {split}")
        if not set(selected_ids) <= split_ids[session_key]:
            raise ValueError(f"selected sessions escape official {split} split")

        for session_id in selected_ids:
            record = by_session.get(session_id)
            if record is None:
                raise ValueError(f"selected session lacks trainval annotation: {session_id}")
            media = media_files.get((split, session_id))
            if media is None:
                raise ValueError(
                    f"selected media file missing from verified manifest: {session_id}"
                )
            media_path = (media_root / split / f"{session_id}.mp4").resolve(strict=True)
            if media_path.stat().st_size != int(media["size_bytes"]):
                raise ValueError(f"media size changed after download: {session_id}")
            actual_hash = sha256_file(media_path)
            if actual_hash != media["sha256"]:
                raise ValueError(f"media SHA-256 changed after download: {session_id}")
            duration = duration_seconds(
                record.get("videoMetadata", {}).get("duration", {}).get("raw")
            )
            if duration is None:
                raise ValueError(f"invalid raw duration for {session_id}")
            media_duration = (
                media_video_durations.get((split, session_id))
                if media_video_durations is not None
                else duration
            )
            if (
                media_duration is None
                or not math.isfinite(float(media_duration))
                or media_duration <= 0
            ):
                raise ValueError(f"missing or invalid verified video duration for {session_id}")

            for event_position, event in enumerate(record.get("events", [])):
                if event.get("label") != "Fine grained action":
                    continue
                action_label = str(event.get("attributes", {}).get("Action Correctness", "UNKNOWN"))
                if action_label == "otherwise":
                    reasons["otherwise_excluded"] += 1
                    continue
                if action_label == "Correct Action":
                    target = 0
                elif action_label in WRONG_LABELS:
                    target = 1
                else:
                    raise ValueError(f"unknown action correctness label: {action_label}")

                start, end = event.get("start"), event.get("end")
                if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
                    reasons["missing_or_non_numeric_interval"] += 1
                    continue
                if end < start:
                    reasons["reversed_interval_excluded"] += 1
                    continue
                if start < 0 or end > media_duration:
                    reasons["out_of_media_interval_excluded"] += 1
                    continue
                event_id = str(event.get("id", event_position))
                example_id = f"{record['taskId']}:{event_id}"
                if example_id in seen_ids:
                    raise ValueError(f"duplicate action example ID: {example_id}")
                seen_ids.add(example_id)
                clip_start = max(0.0, float(start) - pre_context_seconds)
                clip_end = float(end)
                if clip_end <= clip_start:
                    reasons["empty_clip_excluded"] += 1
                    continue
                label_name = "incorrect" if target else "correct"
                label_counts[label_name] += 1
                rows.append(
                    {
                        "id": example_id,
                        "split": split,
                        "session_id": session_id,
                        "media_path": f"{split}/{session_id}.mp4",
                        "media_sha256": actual_hash,
                        "task_type": str(record.get("taskType", "UNKNOWN")),
                        "question": (
                            "For the task ‘"
                            f"{record.get('taskType', 'UNKNOWN')}’, "
                            "does the observed action correctly advance the task?"
                        ),
                        "options": OPTION_TEXT,
                        "target": target,
                        "target_name": label_name,
                        "action_start_seconds": float(start),
                        "action_end_seconds": float(end),
                        "video_duration_seconds": float(media_duration),
                        "annotation_duration_seconds": duration,
                        "clip_start_seconds": clip_start,
                        "clip_end_seconds": clip_end,
                        "future_context_seconds": 0.0,
                    }
                )

        if not rows:
            raise ValueError(f"no valid action events remained in {split}")
        output[split] = rows
        positives = sum(int(row["target"]) for row in rows)
        report[split] = {
            "selected_session_count": len(selected_ids),
            "session_ids_sha256_ordered_newline_delimited": hashlib.sha256(
                "\n".join(selected_ids).encode("utf-8")
            ).hexdigest(),
            "event_count": len(rows),
            "unique_event_id_count": len(seen_ids),
            "correct_count": len(rows) - positives,
            "incorrect_count": positives,
            "incorrect_rate": positives / len(rows),
            "class_counts": dict(sorted(label_counts.items())),
            "excluded_event_counts": dict(sorted(reasons.items())),
        }
    train_sessions = {str(row["session_id"]) for row in output["train"]}
    validation_sessions = {str(row["session_id"]) for row in output["validation"]}
    if train_sessions & validation_sessions:
        raise ValueError("train/validation session leakage in generated event rows")
    return output, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--media-manifest", type=Path, required=True)
    parser.add_argument("--av-validation-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pre-context-seconds", type=float, default=0.5)
    parser.add_argument(
        "--internal-validation-from-train",
        action="store_true",
        help="Use only the official train split; validation is a held-out train partition.",
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite event corpus: {args.output_dir}")
    selection_hash = hashlib.sha256(args.selection.read_bytes()).hexdigest()
    selection = json.loads(args.selection.read_text(encoding="utf-8"))["selection"]
    media_manifest = json.loads(args.media_manifest.read_text(encoding="utf-8"))
    if media_manifest.get("selection_sha256") != selection_hash:
        raise ValueError("media manifest does not match the immutable selection plan")
    media_files = {
        (str(item["split"]), str(item["session_id"])): item
        for item in media_manifest.get("files", [])
    }
    expected_media = {
        (split, str(row["session_id"]))
        for split in ("train", "validation")
        for row in selection[split]["videos"]
    }
    if set(media_files) != expected_media:
        raise ValueError("downloaded media manifest does not exactly match selected sessions")
    media_manifest_hash = sha256_file(args.media_manifest)
    av_validation_report = json.loads(args.av_validation_report.read_text(encoding="utf-8"))
    expected_media_hashes = {
        (split, session_id, str(media_files[(split, session_id)]["sha256"]))
        for split, session_id in expected_media
    }
    validate_av_validation_report(
        av_validation_report,
        selection_sha256=selection_hash,
        media_manifest_sha256=media_manifest_hash,
        expected_media=expected_media_hashes,
    )
    media_video_durations = video_durations_from_av_report(av_validation_report)
    records = json.loads(args.annotation.read_text(encoding="utf-8"))
    if args.internal_validation_from_train:
        train_ids = load_official_train_only(args.splits)
        splits = {"train": train_ids, "validation": train_ids, "test": set()}
    else:
        splits = load_official_splits(args.splits)
    rows, report = build_event_rows(
        records=records,
        split_ids=splits,
        selected_sessions={
            "train": selection["train"]["videos"],
            "validation": selection["validation"]["videos"],
        },
        media_files=media_files,
        media_root=args.media_root,
        media_video_durations=media_video_durations,
        pre_context_seconds=args.pre_context_seconds,
    )
    args.output_dir.mkdir(parents=True)
    for split in ("train", "validation"):
        path = args.output_dir / f"{split}.jsonl"
        with path.open("x", encoding="utf-8", newline="\n") as output:
            for row in rows[split]:
                output.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    receipt = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "selection_sha256": selection_hash,
        "annotation_sha256": sha256_file(args.annotation),
        "official_split_zip_sha256": sha256_file(args.splits),
        "media_manifest_sha256": sha256_file(args.media_manifest),
        "av_validation_report_sha256": sha256_file(args.av_validation_report),
        "media_root": str(args.media_root.resolve(strict=True)),
        "options": OPTION_TEXT,
        "pre_action_context_seconds": args.pre_context_seconds,
        "decision_cutoff": "action interval end; no post-action/future audio or video",
        "report": report,
        "train_jsonl_sha256": sha256_file(args.output_dir / "train.jsonl"),
        "validation_jsonl_sha256": sha256_file(args.output_dir / "validation.jsonl"),
    }
    (args.output_dir / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
