"""Audit annotated event windows against decoded video and audio PTS."""

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

from scripts.audit_perception_av_correspondence import sha256_file
from scripts.audit_perception_av_event_task import (
    TRAIN_SOURCE_SHA256,
    VALID_SOURCE_SHA256,
    load_annotations,
    summarize_split,
)

PLAN_SHA256 = "9ed4f44f3a0cc85465c5a2955706a957326ef36694311c25b4ee51e3c9f18a25"


def load_pinned_media_manifest(manifest_path, split, planned_video_ids):
    """Read one immutable manifest snapshot and return its exact content hash."""
    raw = Path(manifest_path).read_bytes()
    manifest_sha256 = hashlib.sha256(raw).hexdigest()
    manifest = json.loads(raw)
    if manifest.get("split") != split or manifest.get("plan_sha256") != PLAN_SHA256:
        raise ValueError("media manifest split/plan pin mismatch")
    if manifest.get("status") not in {"RUNNING", "PASS_ACQUISITION_NOT_DECODE"}:
        raise ValueError("media acquisition manifest is not in a recognized state")
    clips = manifest.get("clips", [])
    clip_ids = [row.get("video_id") for row in clips]
    if any(not isinstance(video_id, str) for video_id in clip_ids):
        raise ValueError("media manifest contains a missing/invalid video ID")
    if len(clip_ids) != len(set(clip_ids)):
        raise ValueError("media manifest contains duplicate video IDs")
    if not set(clip_ids) <= set(planned_video_ids):
        raise ValueError("media manifest contains unplanned video IDs")
    return manifest, manifest_sha256


def event_window_from_pts(frame_times, frame_durations, frame_interval):
    start_idx, end_idx = frame_interval
    if start_idx < 0 or end_idx < start_idx or end_idx >= len(frame_times):
        raise ValueError("annotated frame interval is outside decoded video")
    start = frame_times[start_idx]
    if end_idx + 1 < len(frame_times):
        end = frame_times[end_idx + 1]
    else:
        duration = frame_durations[end_idx]
        if duration is None or duration <= 0:
            raise ValueError("last annotated frame has no duration for exclusive end")
        end = frame_times[end_idx] + duration
    if not (math.isfinite(start) and math.isfinite(end) and end > start):
        raise ValueError("decoded event PTS window is invalid")
    return start, end


def overlap_samples(event_start, event_end, audio_start, audio_end, sample_rate):
    if sample_rate <= 0 or event_end <= event_start or audio_end <= audio_start:
        raise ValueError("invalid audio/event time interval")
    start = max(event_start, audio_start)
    end = min(event_end, audio_end)
    if end <= start:
        return 0, 0, 0.0
    start_sample = round((start - audio_start) * sample_rate)
    end_sample = round((end - audio_start) * sample_rate)
    if end_sample < start_sample:
        raise ValueError("audio sample mapping is not monotonic")
    return start_sample, end_sample, (end - start) / (event_end - event_start)


def decode_timing(path):
    import av

    frame_times = []
    frame_durations = []
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError("source clip must contain both video and audio")
        for frame in container.decode(video=0):
            if frame.time is None or not math.isfinite(frame.time):
                raise ValueError("missing/non-finite decoded video PTS")
            stamp = float(frame.time)
            if frame_times and stamp <= frame_times[-1]:
                raise ValueError("video PTS are not strictly increasing")
            frame_times.append(stamp)
            duration = None
            if frame.duration is not None and frame.time_base is not None:
                duration = float(frame.duration * frame.time_base)
            frame_durations.append(duration)
    with av.open(str(path)) as container:
        if not container.streams.audio:
            raise ValueError("source clip must contain both video and audio")
        audio_start = None
        audio_end = None
        audio_rate = None
        decoded_samples = 0
        for frame in container.decode(audio=0):
            if frame.pts is None or frame.sample_rate <= 0:
                raise ValueError("decoded audio frame lacks PTS/sample rate")
            stamp = float(frame.pts * frame.time_base)
            if audio_rate is not None and frame.sample_rate != audio_rate:
                raise ValueError("audio sample rate changed within clip")
            audio_rate = frame.sample_rate
            if audio_start is None:
                audio_start = stamp
            if audio_end is not None and abs(stamp - audio_end) > 1.0 / audio_rate:
                raise ValueError("decoded audio PTS contain a gap or overlap")
            audio_end = stamp + frame.samples / audio_rate
            decoded_samples += frame.samples
    if not frame_times or audio_start is None or audio_end is None:
        raise ValueError("empty decoded audio/video stream")
    return {
        "pyav_version": av.__version__,
        "video_frame_count": len(frame_times),
        "video_frame_times": frame_times,
        "video_frame_durations": frame_durations,
        "audio_start_seconds": audio_start,
        "audio_end_seconds": audio_end,
        "audio_sample_rate": audio_rate,
        "audio_decoded_samples": decoded_samples,
    }


def audit_split(split, source_root, media_dir, plan):
    import av

    source_root = Path(source_root)
    media_dir = Path(media_dir)
    source_pins = {"train": TRAIN_SOURCE_SHA256, "valid": VALID_SOURCE_SHA256}[split]
    annotation_path = source_root / f"{split}_annotations.zip"
    cut_path = source_root / f"cut_frame_mapping_{split}.json"
    data, cuts, source_hashes = load_annotations(
        annotation_path,
        cut_path,
        split,
        {"annotations": source_pins["annotations"], "cuts": source_pins["cuts"]},
    )
    selected_rows = {row["video_id"]: row for row in plan["splits"][split]["rows"]}
    events_by_video = defaultdict(list)
    for event in summarize_split(data, cuts, split)["candidate_records"]:
        if event["video_id"] in selected_rows:
            events_by_video[event["video_id"]].append(event)
    manifest_path = media_dir / "manifest.json"
    manifest, manifest_sha256 = load_pinned_media_manifest(
        manifest_path, split, selected_rows
    )
    clips = {row["video_id"]: row for row in manifest.get("clips", [])}
    records = []
    missing_media = sorted(set(events_by_video) - set(clips))
    for video_id in sorted(set(events_by_video) & set(clips)):
        clip = clips[video_id]
        path = Path(clip["path"])
        if not path.is_file() or path.stat().st_size != clip["bytes"]:
            raise ValueError(f"missing or truncated acquired media: {video_id}")
        media_sha = sha256_file(path)
        if media_sha != clip.get("sha256"):
            raise ValueError(f"acquired media hash mismatch: {video_id}")
        timing = decode_timing(path)
        metadata = selected_rows[video_id]["metadata"]
        for event in events_by_video[video_id]:
            start, end = event_window_from_pts(
                timing["video_frame_times"],
                timing["video_frame_durations"],
                event["frame_interval"],
            )
            sample_start, sample_end, coverage = overlap_samples(
                start,
                end,
                timing["audio_start_seconds"],
                timing["audio_end_seconds"],
                timing["audio_sample_rate"],
            )
            records.append(
                {
                    "video_id": video_id,
                    "sound_id": event["sound_id"],
                    "frame_interval": event["frame_interval"],
                    "selected_frame": event["selected_frame"],
                    "candidate_count": len(event["candidate_track_ids"]),
                    "event_start_seconds": start,
                    "event_end_seconds": end,
                    "audio_sample_start": sample_start,
                    "audio_sample_end_exclusive": sample_end,
                    "audio_coverage_fraction": coverage,
                    "media_sha256": media_sha,
                    "decoded_video_frames": timing["video_frame_count"],
                    "metadata_video_frames": metadata["num_frames"],
                    "video_frame_count_delta": (
                        timing["video_frame_count"] - metadata["num_frames"]
                    ),
                    "audio_decoded_samples": timing["audio_decoded_samples"],
                    "metadata_audio_samples": metadata["audio_samples"],
                    "decoded_audio_sample_rate": timing["audio_sample_rate"],
                    "metadata_audio_sample_rate": metadata["audio_sample_rate"],
                    "audio_sample_rate_mismatch": (
                        timing["audio_sample_rate"] != metadata["audio_sample_rate"]
                    ),
                    "audio_sample_count_delta": (
                        timing["audio_decoded_samples"] - metadata["audio_samples"]
                    ),
                }
            )
    expected_events = sum(map(len, events_by_video.values()))
    accounted_events = len(records) + sum(
        len(events_by_video[video_id]) for video_id in missing_media
    )
    if accounted_events != expected_events:
        raise ValueError("event accounting mismatch")
    if not missing_media and manifest["status"] != "PASS_ACQUISITION_NOT_DECODE":
        raise ValueError("complete media inventory lacks terminal acquisition receipt")
    return {
        "split": split,
        "status": manifest["status"],
        "source_hashes": source_hashes,
        "media_manifest_sha256": manifest_sha256,
        "acquired_videos": len(clips),
        "selected_videos": len(selected_rows),
        "decoded_videos": len(set(record["video_id"] for record in records)),
        "expected_events": expected_events,
        "decoded_events": len(records),
        "missing_media_videos": missing_media,
        "zero_audio_overlap_events": sum(
            record["audio_coverage_fraction"] == 0 for record in records
        ),
        "partial_audio_overlap_events": sum(
            record["audio_coverage_fraction"] < 1 for record in records
        ),
        "minimum_audio_coverage_fraction": min(
            (record["audio_coverage_fraction"] for record in records), default=None
        ),
        "video_frame_count_mismatch_videos": len(
            {
                record["video_id"]
                for record in records
                if record["video_frame_count_delta"] != 0
            }
        ),
        "audio_sample_count_mismatch_videos": len(
            {
                record["video_id"]
                for record in records
                if record["audio_sample_count_delta"] != 0
            }
        ),
        "audio_sample_rate_mismatch_videos": len(
            {
                record["video_id"]
                for record in records
                if record["audio_sample_rate_mismatch"]
            }
        ),
        "pyav_version": av.__version__,
        "events": records,
    }


def audit(plan_path, train_source, valid_source, train_media, valid_media, allow_partial=False):
    if sha256_file(plan_path) != PLAN_SHA256:
        raise ValueError("media plan hash changed")
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    if plan.get("status") != "PASS_FRESH_EVENT_MEDIA_PLAN" or plan.get("test_opened"):
        raise ValueError("expected pinned train/valid-only media plan")
    splits = {
        "train": audit_split("train", train_source, train_media, plan),
        "valid": audit_split("valid", valid_source, valid_media, plan),
    }
    missing = sum(len(result["missing_media_videos"]) for result in splits.values())
    if missing and not allow_partial:
        raise ValueError(f"media acquisition incomplete for {missing} videos")
    return {
        "status": "PASS_PARTIAL_PTS_AUDIT" if missing else "PASS_COMPLETE_PTS_AUDIT",
        "plan_sha256": PLAN_SHA256,
        "test_opened": False,
        "model_loaded": False,
        "training": False,
        "gpu_used": False,
        "splits": splits,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--train-source", type=Path, required=True)
    parser.add_argument("--valid-source", type=Path, required=True)
    parser.add_argument("--train-media", type=Path, required=True)
    parser.add_argument("--valid-media", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("fresh output path required")
    result = audit(
        args.plan,
        args.train_source,
        args.valid_source,
        args.train_media,
        args.valid_media,
        allow_partial=args.allow_partial,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                split: {
                    key: value for key, value in row.items() if key != "events"
                }
                for split, row in result["splits"].items()
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
