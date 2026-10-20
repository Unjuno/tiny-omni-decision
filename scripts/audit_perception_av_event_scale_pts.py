"""Decode and audit PTS/audio coverage for the frozen 388-video scale cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path

from scripts.acquire_perception_av_event_fresh_holdout import previous_media_inventory
from scripts.audit_perception_av_correspondence import sha256_file
from scripts.audit_perception_av_event_media_pts import (
    decode_timing,
    event_window_from_pts_with_single_frame_tail_censor,
    overlap_samples,
)
from scripts.audit_perception_av_event_task import load_annotations, summarize_split
from scripts.plan_perception_av_event_fresh_holdout import ANNOTATION_SHA256

PLAN_SHA256 = "43dda454adf45352dee281680f51698ecf38d60731d33311a6ded139de167461"
HOLDOUT_MANIFEST_SHA256 = "8591dbb0c173c70780c6d6066f18b2d7e3f29ba43c215ae0b2a251a07e2ac118"
ROOT = Path("C:/CodexArtifacts/pretrained-reuse-r2")


def audit(
    plan_path: Path,
    media_dir: Path,
    source_dir: Path,
    output: Path,
    root: Path = ROOT,
) -> dict:
    plan_path = Path(plan_path)
    plan_hash = sha256_file(plan_path)
    if plan_hash != PLAN_SHA256:
        raise ValueError("scale plan hash differs from the preregistered plan")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("status") != "PASS_FRESH_VIDEO_SCALE_PLAN" or plan.get("test_opened") is not False:
        raise ValueError("expected a test-closed scale plan")

    manifest_path = Path(media_dir) / "manifest.json"
    raw_manifest = manifest_path.read_bytes()
    manifest_hash = hashlib.sha256(raw_manifest).hexdigest()
    manifest = json.loads(raw_manifest)
    if (
        manifest.get("status") != "PASS_ACQUIRED_UNVERIFIED_DECODE"
        or manifest.get("plan_sha256") != plan_hash
        or manifest.get("training") is not False
        or manifest.get("test_opened") is not False
        or len(manifest.get("clips", [])) != 388
    ):
        raise ValueError("complete acquisition receipt is not pinned to the scale plan")

    planned = {row["video_id"]: row for row in plan["rows"]}
    clips = {row["video_id"]: row for row in manifest["clips"]}
    if len(clips) != 388 or planned.keys() != clips.keys():
        raise ValueError("acquired media inventory differs from the frozen plan")
    if any(clips[key].get("model_split") != planned[key].get("model_split") for key in planned):
        raise ValueError("acquired video split differs from the plan")

    prior_hashes, prior_ids = previous_media_inventory(plan, Path(root))
    holdout_manifest_path = (
        Path(root) / "perception-av-event-fresh-holdout-media-01" / "manifest.json"
    )
    if sha256_file(holdout_manifest_path) != HOLDOUT_MANIFEST_SHA256:
        raise ValueError("fixed-checkpoint holdout media manifest changed")
    holdout = json.loads(holdout_manifest_path.read_text(encoding="utf-8"))
    if holdout.get("status") != "PASS_ACQUIRED_UNVERIFIED_DECODE":
        raise ValueError("fixed-checkpoint holdout acquisition is incomplete")
    for clip in holdout["clips"]:
        prior_ids.add(clip["video_id"])
        prior_hashes.add(clip["sha256"])
    if set(planned) & prior_ids:
        raise ValueError("scale videos overlap an earlier media identity")

    annotations, cuts, source_hashes = load_annotations(
        Path(source_dir) / "train_annotations.zip",
        Path(source_dir) / "cut_frame_mapping_train.json",
        "train",
        {
            "annotations": ANNOTATION_SHA256["train_annotations.zip"],
            "cuts": ANNOTATION_SHA256["cut_frame_mapping_train.json"],
        },
    )
    events_by_video = defaultdict(list)
    for event in summarize_split(annotations, cuts, "train")["candidate_records"]:
        if event["video_id"] in planned:
            events_by_video[event["video_id"]].append(event)
    if set(events_by_video) != set(planned):
        raise ValueError("selected video set differs from eligible candidate-event videos")

    records = []
    seen_hashes = set(prior_hashes)
    for index, video_id in enumerate(sorted(planned), start=1):
        clip = clips[video_id]
        path = Path(clip["path"])
        if not path.is_file() or path.stat().st_size != clip.get("bytes"):
            raise ValueError(f"missing or truncated media: {video_id}")
        media_hash = sha256_file(path)
        if media_hash != clip.get("sha256") or media_hash in seen_hashes:
            raise ValueError(f"media hash mismatch or duplicate identity: {video_id}")
        seen_hashes.add(media_hash)
        timing = decode_timing(path)
        for event in events_by_video[video_id]:
            start, end, right_censored = event_window_from_pts_with_single_frame_tail_censor(
                timing["video_frame_times"],
                timing["video_frame_durations"],
                event["frame_interval"],
            )
            if event["selected_frame"] >= timing["video_frame_count"]:
                raise ValueError(f"selected frame is missing: {video_id}/{event['sound_id']}")
            left, right, coverage = overlap_samples(
                start,
                end,
                timing["audio_start_seconds"],
                timing["audio_end_seconds"],
                timing["audio_sample_rate"],
            )
            if coverage < 1.0 or right <= left:
                raise ValueError(
                    f"event audio lacks full PTS coverage: {video_id}/{event['sound_id']}"
                )
            records.append(
                {
                    "video_id": video_id,
                    "model_split": planned[video_id]["model_split"],
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
            print(f"decoded {index}/388 videos; {len(records)} events", flush=True)

    expected_events = sum(row["candidate_events"] for row in planned.values())
    if len(records) != expected_events:
        raise ValueError("decoded event count differs from plan")
    split_counts = {
        split: sum(row["model_split"] == split for row in records)
        for split in ("fit", "selection", "calibration", "development_validation")
    }
    expected_by_split = {
        split: plan["split_summary"][split]["candidate_events"] for split in split_counts
    }
    if split_counts != expected_by_split:
        raise ValueError("decoded per-split event counts differ from plan")

    result = {
        "status": "PASS_COMPLETE_VIDEO_SCALE_PTS_AUDIT",
        "plan_sha256": plan_hash,
        "media_manifest_sha256": manifest_hash,
        "audit_script_sha256": sha256_file(Path(__file__)),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_hashes": source_hashes,
        "video_count": len(clips),
        "split_video_counts": {
            split: plan["split_summary"][split]["video_count"] for split in split_counts
        },
        "split_event_counts": split_counts,
        "unique_media_hashes_across_prior_and_current": len(seen_hashes),
        "event_count": len(records),
        "right_censored_events": sum(row["right_censored_at_media_end"] for row in records),
        "zero_or_partial_audio_coverage_events": sum(
            row["audio_coverage_fraction"] < 1.0 for row in records
        ),
        "pyav_version": __import__("av").__version__,
        "gpu_used": False,
        "training": False,
        "test_opened": False,
        "events": records,
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
    args = parser.parse_args()
    result = audit(args.plan, args.media, args.source, args.output)
    print(
        json.dumps({key: value for key, value in result.items() if key != "events"}, sort_keys=True)
    )


if __name__ == "__main__":
    main()
