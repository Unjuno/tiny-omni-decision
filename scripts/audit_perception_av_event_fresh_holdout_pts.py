"""CPU decode and audit actual PTS for the fixed fresh TRAIN holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path

from scripts.acquire_perception_av_event_fresh_holdout import (
    FRESH_PLAN_SHA256,
    previous_media_inventory,
)
from scripts.audit_perception_av_correspondence import sha256_file
from scripts.audit_perception_av_event_media_pts import (
    decode_timing,
    event_window_from_pts_with_single_frame_tail_censor,
    overlap_samples,
)
from scripts.audit_perception_av_event_task import load_annotations, summarize_split
from scripts.plan_perception_av_event_fresh_holdout import (
    ANNOTATION_SHA256,
    ROOT,
)

PINNED_PLAN_SHA256 = FRESH_PLAN_SHA256


def audit(
    plan_path: Path,
    media_dir: Path,
    source_dir: Path,
    output: Path,
    allow_partial: bool = False,
) -> dict:
    plan_path = Path(plan_path)
    plan_sha256 = sha256_file(plan_path)
    if plan_sha256 != PINNED_PLAN_SHA256:
        raise ValueError("fresh holdout plan hash differs from the preregistered plan")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("status") != "PASS_FRESH_TRAIN_HOLDOUT_PLAN" or plan.get("test_opened"):
        raise ValueError("expected test-closed fresh TRAIN holdout plan")
    if len(plan.get("rows", [])) != 200:
        raise ValueError("expected exactly 200 fixed holdout videos")
    media_manifest_path = Path(media_dir) / "manifest.json"
    media_manifest_raw = media_manifest_path.read_bytes()
    media_manifest = json.loads(media_manifest_raw)
    manifest_hash = hashlib.sha256(media_manifest_raw).hexdigest()
    complete_acquisition = media_manifest.get("status") == "PASS_ACQUIRED_UNVERIFIED_DECODE"
    if (
        media_manifest.get("status") not in {"RUNNING", "PASS_ACQUIRED_UNVERIFIED_DECODE"}
        or media_manifest.get("plan_sha256") != sha256_file(plan_path)
        or media_manifest.get("test_opened") is not False
        or (not complete_acquisition and not allow_partial)
        or (complete_acquisition and len(media_manifest.get("clips", [])) != 200)
    ):
        raise ValueError("acquisition receipt is not pinned to this plan or permitted extent")
    data, cuts, source_hashes = load_annotations(
        Path(source_dir) / "train_annotations.zip",
        Path(source_dir) / "cut_frame_mapping_train.json",
        "train",
        {
            "annotations": ANNOTATION_SHA256["train_annotations.zip"],
            "cuts": ANNOTATION_SHA256["cut_frame_mapping_train.json"],
        },
    )
    planned = {row["video_id"]: row for row in plan["rows"]}
    clips = {row["video_id"]: row for row in media_manifest["clips"]}
    if not clips or not clips.keys() <= planned.keys():
        raise ValueError("acquired media IDs are empty or differ from the frozen plan")
    prior_hashes, prior_video_ids = previous_media_inventory(plan, ROOT)
    if set(clips) & prior_video_ids:
        raise ValueError("holdout media IDs overlap previously acquired videos")
    if not allow_partial and planned.keys() != clips.keys():
        raise ValueError("complete media inventory differs from the frozen plan")
    events_by_video = defaultdict(list)
    for event in summarize_split(data, cuts, "train")["candidate_records"]:
        if event["video_id"] in clips:
            events_by_video[event["video_id"]].append(event)
    if set(events_by_video) != set(clips):
        raise ValueError("acquired media and eligible-event videos differ")

    event_rows = []
    seen_media_hashes = set(prior_hashes)
    for index, video_id in enumerate(sorted(clips), start=1):
        clip = clips[video_id]
        path = Path(clip["path"])
        if not path.is_file() or path.stat().st_size != clip["bytes"]:
            raise ValueError(f"missing or truncated media: {video_id}")
        media_hash = sha256_file(path)
        if media_hash != clip.get("sha256"):
            raise ValueError(f"media SHA-256 mismatch: {video_id}")
        if media_hash in seen_media_hashes:
            raise ValueError(f"exact media-byte duplicate across prior/current data: {video_id}")
        seen_media_hashes.add(media_hash)
        timing = decode_timing(path)
        for event in events_by_video[video_id]:
            start, end, right_censored = event_window_from_pts_with_single_frame_tail_censor(
                timing["video_frame_times"],
                timing["video_frame_durations"],
                event["frame_interval"],
            )
            if event["selected_frame"] >= timing["video_frame_count"]:
                raise ValueError(
                    f"selected candidate frame missing: {video_id}/{event['sound_id']}"
                )
            left, right, coverage = overlap_samples(
                start,
                end,
                timing["audio_start_seconds"],
                timing["audio_end_seconds"],
                timing["audio_sample_rate"],
            )
            if coverage < 1.0 or right <= left:
                raise ValueError(
                    f"event audio is not fully covered: {video_id}/{event['sound_id']}"
                )
            event_rows.append(
                {
                    "video_id": video_id,
                    "sound_id": event["sound_id"],
                    "frame_interval": event["frame_interval"],
                    "selected_frame": event["selected_frame"],
                    "candidate_count": len(event["candidate_track_ids"]),
                    "event_start_seconds": start,
                    "event_end_seconds": end,
                    "audio_sample_start": left,
                    "audio_sample_end_exclusive": right,
                    "audio_sample_rate": timing["audio_sample_rate"],
                    "right_censored_at_media_end": right_censored,
                    "audio_coverage_fraction": coverage,
                    "media_sha256": media_hash,
                }
            )
        if index % 10 == 0:
            print(f"decoded {index}/{len(clips)} videos; {len(event_rows)} events", flush=True)

    expected_events = sum(planned[video_id]["candidate_events"] for video_id in clips)
    if len(event_rows) != expected_events:
        raise ValueError("decoded event count differs from acquired plan subset")
    is_complete = planned.keys() == clips.keys() and complete_acquisition
    result = {
        "status": (
            "PASS_COMPLETE_FRESH_HOLDOUT_PTS_AUDIT"
            if is_complete
            else "PASS_PARTIAL_FRESH_HOLDOUT_PTS_SMOKE"
        ),
        "plan_sha256": plan_sha256,
        "media_manifest_sha256": manifest_hash,
        "audit_script_sha256": sha256_file(Path(__file__)),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_hashes": source_hashes,
        "video_count": len(clips),
        "planned_video_count": len(planned),
        "unique_media_hashes_across_prior_and_current": len(seen_media_hashes),
        "event_count": len(event_rows),
        "right_censored_events": sum(row["right_censored_at_media_end"] for row in event_rows),
        "zero_or_partial_audio_coverage_events": sum(
            row["audio_coverage_fraction"] < 1.0 for row in event_rows
        ),
        "pyav_version": __import__("av").__version__,
        "gpu_used": False,
        "training": False,
        "test_opened": False,
        "events": event_rows,
    }
    output = Path(output)
    if output.exists():
        raise ValueError("fresh PTS audit output required")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--media", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=ROOT / "perception-test-source-audit-01")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    result = audit(args.plan, args.media, args.source, args.output, args.allow_partial)
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "events"}, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
