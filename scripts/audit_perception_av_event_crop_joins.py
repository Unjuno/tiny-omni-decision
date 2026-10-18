"""Validate pinned event-to-track box joins without loading a model or labels as inputs."""

import argparse
import json
from collections import Counter
from pathlib import Path

from scripts.audit_perception_av_correspondence import sha256_file
from scripts.audit_perception_av_event_task import (
    TRAIN_SOURCE_SHA256,
    VALID_SOURCE_SHA256,
    load_annotations,
    summarize_split,
)
from scripts.audit_perception_av_event_vision_smoke import (
    normalized_box_to_pixels,
    track_box_at_frame,
)
from scripts.perception_av_event_protocol import (
    deterministic_candidate_order,
    model_selection_video_ids,
)

PLAN_SHA256 = "9ed4f44f3a0cc85465c5a2955706a957326ef36694311c25b4ee51e3c9f18a25"


def validate_event_crop_join(event, tracks_by_id, resolution):
    if len(resolution) != 2:
        raise ValueError("annotation resolution must be height and width")
    height, width = resolution
    if type(height) is not int or type(width) is not int or min(height, width) < 1:
        raise ValueError("annotation resolution must contain positive integers")
    frame_interval = event.get("frame_interval")
    frame_id = event.get("selected_frame")
    if (
        not isinstance(frame_interval, list)
        or len(frame_interval) != 2
        or any(type(value) is not int for value in frame_interval)
        or frame_interval[0] < 0
        or frame_interval[1] < frame_interval[0]
        or type(frame_id) is not int
        or not frame_interval[0] <= frame_id <= frame_interval[1]
    ):
        raise ValueError("selected frame must lie inside the eligible event interval")
    candidate_ids = event.get("candidate_track_ids")
    source_ids = event.get("source_parent_ids")
    if (
        not isinstance(candidate_ids, list)
        or len(candidate_ids) < 2
        or len(candidate_ids) != len(set(candidate_ids))
        or not isinstance(source_ids, list)
        or len(source_ids) != 1
        or source_ids[0] not in candidate_ids
    ):
        raise ValueError("eligible event must have unique candidates and one included source")
    ordered_ids = deterministic_candidate_order(
        event["video_id"], event["sound_id"], candidate_ids
    )
    pixel_boxes = []
    for track_id in ordered_ids:
        track = tracks_by_id.get(track_id)
        if track is None:
            raise ValueError("candidate track ID is absent from annotation track table")
        box = track_box_at_frame(track, frame_id)
        pixel_boxes.append(normalized_box_to_pixels(box, height, width))
    return {
        "video_id": event["video_id"],
        "sound_id": event["sound_id"],
        "selected_frame": frame_id,
        "frame_interval": frame_interval,
        "candidate_count": len(ordered_ids),
        "source_candidate_index_for_separate_label_join": ordered_ids.index(source_ids[0]),
        "pixel_boxes_in_model_input_order": pixel_boxes,
    }


def group_planned_events(candidate_records, planned_video_ids):
    planned_list = list(planned_video_ids)
    planned_ids = set(planned_list)
    if len(planned_ids) != len(planned_list):
        raise ValueError("planned video IDs must be unique")
    by_video = {}
    for event in candidate_records:
        if event["video_id"] in planned_ids:
            by_video.setdefault(event["video_id"], []).append(event)
    return by_video


def audit_split(split, source_root, media_dir, plan):
    pins = {"train": TRAIN_SOURCE_SHA256, "valid": VALID_SOURCE_SHA256}[split]
    source_root, media_dir = Path(source_root), Path(media_dir)
    data, cuts, source_hashes = load_annotations(
        source_root / f"{split}_annotations.zip",
        source_root / f"cut_frame_mapping_{split}.json",
        split,
        {"annotations": pins["annotations"], "cuts": pins["cuts"]},
    )
    plan_rows = plan["splits"][split]["rows"]
    planned = {row["video_id"]: row for row in plan_rows}
    if len(planned) != len(plan_rows):
        raise ValueError(f"duplicate planned {split} video ID")
    manifest_path = media_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("split") != split
        or manifest.get("plan_sha256") != PLAN_SHA256
        or manifest.get("status") not in {"RUNNING", "PASS_ACQUISITION_NOT_DECODE"}
    ):
        raise ValueError(f"{split} media manifest does not match the pinned plan")
    clips = {row["video_id"]: row for row in manifest.get("clips", [])}
    if len(clips) != len(manifest.get("clips", [])) or not set(clips) <= set(planned):
        raise ValueError(f"{split} acquired clips contain duplicate or unplanned IDs")

    events = summarize_split(data, cuts, split)["candidate_records"]
    by_video = group_planned_events(events, planned)
    plan_event_count = sum(row["candidate_events"] for row in plan_rows)
    if sum(map(len, by_video.values())) != plan_event_count:
        raise ValueError(f"{split} selected-event count differs from frozen media plan")
    validated = []
    event_counts = Counter()
    for video_id, clip in sorted(clips.items()):
        media = Path(clip["path"])
        if not media.is_file() or media.stat().st_size != clip.get("bytes"):
            raise ValueError(f"acquired {split} media is missing/truncated: {video_id}")
        if sha256_file(media) != clip.get("sha256"):
            raise ValueError(f"acquired {split} media hash mismatch: {video_id}")
        annotation = data[video_id]
        if annotation["metadata"]["resolution"] != clip["metadata"]["resolution"]:
            raise ValueError(f"{split} media/annotation resolution mismatch: {video_id}")
        tracks = {track["id"]: track for track in annotation["object_tracking"]}
        for event in by_video.get(video_id, []):
            validated.append(
                validate_event_crop_join(event, tracks, annotation["metadata"]["resolution"])
            )
            event_counts[len(event["candidate_track_ids"])] += 1
    all_events = sum(map(len, by_video.values()))
    return {
        "split": split,
        "status": manifest["status"],
        "source_hashes": source_hashes,
        "media_manifest_sha256": sha256_file(manifest_path),
        "planned_videos": len(planned),
        "acquired_videos": len(clips),
        "acquired_videos_with_eligible_events": len(set(row["video_id"] for row in validated)),
        "eligible_events_total": all_events,
        "eligible_events_joined": len(validated),
        "eligible_events_waiting_for_media": all_events - len(validated),
        "candidate_count_distribution_joined": dict(sorted(event_counts.items())),
        "events": validated,
    }


def audit(plan_path, train_source, valid_source, train_media, valid_media):
    if sha256_file(plan_path) != PLAN_SHA256:
        raise ValueError("media plan SHA-256 does not match the pinned revision")
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    if plan.get("status") != "PASS_FRESH_EVENT_MEDIA_PLAN":
        raise ValueError("unexpected media plan status")
    train_ids = {row["video_id"] for row in plan["splits"]["train"]["rows"]}
    valid_ids = {row["video_id"] for row in plan["splits"]["valid"]["rows"]}
    if train_ids & valid_ids:
        raise ValueError("TRAIN and VALID video IDs overlap")
    train_rows = plan["splits"]["train"]["rows"]
    fit_ids = {row["video_id"] for row in train_rows if row.get("model_split") == "train"}
    calibration_ids = {
        row["video_id"]
        for row in train_rows
        if row.get("model_split") == "calibration"
    }
    if (
        fit_ids | calibration_ids != train_ids
        or fit_ids & calibration_ids
    ):
        raise ValueError("TRAIN fit/calibration groups do not partition TRAIN videos")
    if len(train_ids) != 400 or len(fit_ids) != 300 or len(calibration_ids) != 100:
        raise ValueError("TRAIN fit/calibration group counts differ from the frozen plan")
    selection_ids = model_selection_video_ids(sorted(fit_ids))
    if len(selection_ids) != 60 or selection_ids - fit_ids:
        raise ValueError("nested selection split escaped TRAIN fit pool")
    selection_partition = {
        "train_fit_videos": len(fit_ids) - len(selection_ids),
        "train_selection_videos": len(selection_ids),
        "train_fit_pool_videos": len(fit_ids),
        "train_calibration_videos": len(calibration_ids),
        "development_validation_videos": len(valid_ids),
    }
    train_result = audit_split("train", train_source, train_media, plan)
    valid_result = audit_split("valid", valid_source, valid_media, plan)
    complete = all(
        result["status"] == "PASS_ACQUISITION_NOT_DECODE"
        and result["eligible_events_waiting_for_media"] == 0
        for result in (train_result, valid_result)
    )
    return {
        "status": "PASS_COMPLETE_CROP_JOIN_AUDIT" if complete else "PASS_PARTIAL_CROP_JOIN_AUDIT",
        "plan_sha256": PLAN_SHA256,
        "video_group_partition": selection_partition,
        "training": False,
        "model_loaded": False,
        "quality_evaluated": False,
        "gpu_used": False,
        "test_opened": False,
        "splits": {"train": train_result, "valid": valid_result},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--train-source", type=Path, required=True)
    parser.add_argument("--valid-source", type=Path, required=True)
    parser.add_argument("--train-media", type=Path, required=True)
    parser.add_argument("--valid-media", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("fresh output file required")
    result = audit(
        args.plan,
        args.train_source,
        args.valid_source,
        args.train_media,
        args.valid_media,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {key: value for key, value in result.items() if key != "splits"}
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
