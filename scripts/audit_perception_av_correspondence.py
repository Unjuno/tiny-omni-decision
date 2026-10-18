"""Train-only feasibility audit for audio-event to visible-track decisions."""

import argparse
import collections
import hashlib
import json
import math
import platform
import zipfile
from pathlib import Path

SOURCE_REVISION = "3938d2f1ba3a6b502025741cea4cd73c7b3bdfaf"
TRAIN_ARCHIVE_SHA256 = "9eaa6aad3fabdf4cde4955f333cfddf3b9e6839c5abcb271afa410870f640d78"
CUT_MAP_SHA256 = "cbea6c721fd150f861ec6fe8e7ff005792f754df1359b1364349b5abcaab6cc4"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def event_candidate(sound, tracks, cut_frame):
    """Classify one sound segment without turning unknown or multi-source labels into negatives."""
    parents = sound.get("parent_objects")
    by_id = {track.get("id"): track for track in tracks}
    if not parents or any(
        type(parent) is not int or parent < 0 or parent not in by_id for parent in parents
    ):
        reason = "unknown_or_unresolved_parent"
    elif sound.get("label", "").strip() == "Muted":
        reason = "muted_audio_label"
    elif sound.get("is_visible") != 1:
        reason = "source_not_visible"
    elif len(parents) != 1:
        reason = "multiple_parent_sources"
    else:
        frame_ids = sound.get("frame_ids")
        if (
            not isinstance(frame_ids, list)
            or len(frame_ids) != 2
            or any(type(frame) is not int for frame in frame_ids)
            or frame_ids[0] < 0
            or frame_ids[1] < frame_ids[0]
        ):
            raise ValueError("invalid sound frame interval")
        start, end = frame_ids
        if cut_frame >= 0 and end >= cut_frame:
            reason = "crosses_or_reaches_cut"
        else:
            visible_by_frame = collections.defaultdict(set)
            for track in tracks:
                ids, boxes, masked = (
                    track.get("frame_ids"),
                    track.get("bounding_boxes"),
                    track.get("is_masked"),
                )
                if (
                    not isinstance(ids, list)
                    or not isinstance(boxes, list)
                    or not isinstance(masked, list)
                ):
                    raise ValueError("track annotations must be lists")
                if not (len(ids) == len(boxes) == len(masked)):
                    raise ValueError("track annotation lengths differ")
                for frame, box, is_masked in zip(ids, boxes, masked, strict=True):
                    if (
                        type(frame) is not int
                        or type(is_masked) is not int
                        or is_masked not in (0, 1)
                    ):
                        raise ValueError("invalid track frame or visibility flag")
                    if frame < start or frame > end or (cut_frame >= 0 and frame >= cut_frame):
                        continue
                    if (
                        not isinstance(box, list)
                        or len(box) != 4
                        or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in box)
                    ):
                        raise ValueError("invalid track box")
                    x0, y0, x1, y1 = box
                    if is_masked == 0 and x1 > x0 and y1 > y0:
                        visible_by_frame[frame].add(track["id"])
            competitor_frames = [
                frame for frame, visible_ids in visible_by_frame.items() if len(visible_ids) >= 2
            ]
            if not competitor_frames:
                reason = "no_shared_visible_competitor_frame"
            else:
                midpoint = (start + end) / 2
                selected_frame = min(
                    competitor_frames, key=lambda frame: (abs(frame - midpoint), frame)
                )
                candidate_track_ids = sorted(visible_by_frame[selected_frame])
                if parents[0] not in candidate_track_ids:
                    reason = "source_absent_at_fixed_frame"
                else:
                    return {
                        "status": "candidate",
                        "reason": None,
                        "selected_frame": selected_frame,
                        "candidate_track_ids": candidate_track_ids,
                    }
    return {
        "status": "held",
        "reason": reason,
        "selected_frame": None,
        "candidate_track_ids": [],
    }


def summarize(data, cuts):
    counts = collections.Counter()
    by_label = collections.Counter()
    option_counts = collections.Counter()
    candidate_videos = set()
    records = []
    for video_id, row in sorted(data.items()):
        metadata = row.get("metadata", {})
        if metadata.get("split") != "train" or metadata.get("video_id") != video_id:
            raise ValueError("train-only video identity required")
        if video_id not in cuts or type(cuts[video_id]) is not int:
            raise ValueError("pinned cut mapping required for every train video")
        tracks = row.get("object_tracking", [])
        track_ids = [track.get("id") for track in tracks]
        if any(type(track_id) is not int or track_id < 0 for track_id in track_ids):
            raise ValueError("object track IDs must be nonnegative integers")
        if len(track_ids) != len(set(track_ids)):
            raise ValueError("duplicate object track ID")
        for sound in row.get("sound_localisation", []):
            decision = event_candidate(sound, tracks, cuts[video_id])
            counts["sound_segments"] += 1
            counts[decision["status"]] += 1
            if decision["reason"]:
                counts["held:" + decision["reason"]] += 1
            if decision["status"] == "candidate":
                candidate_videos.add(video_id)
                by_label[sound["label"]] += 1
                option_counts[len(decision["candidate_track_ids"])] += 1
            records.append(
                {
                    "video_id": video_id,
                    "sound_id": sound["id"],
                    "sound_label": sound["label"],
                    "source_parent_ids": list(sound["parent_objects"]),
                    "frame_interval": list(sound["frame_ids"]),
                    **decision,
                }
            )
    if counts["candidate"] + counts["held"] != counts["sound_segments"]:
        raise ValueError("lost sound annotations")
    return {
        "counts": dict(sorted(counts.items())),
        "candidate_video_count": len(candidate_videos),
        "eligible_by_sound_label": dict(sorted(by_label.items())),
        "option_count_distribution": dict(sorted(option_counts.items())),
        "records": records,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("fresh output directory required")
    archive = args.source / "train_annotations.zip"
    cut_path = args.source / "cut_frame_mapping_train.json"
    pins = {
        "train_annotations.zip": TRAIN_ARCHIVE_SHA256,
        "cut_frame_mapping_train.json": CUT_MAP_SHA256,
    }
    if sha256_file(archive) != TRAIN_ARCHIVE_SHA256 or sha256_file(cut_path) != CUT_MAP_SHA256:
        raise ValueError("unreviewed Perception Test source bytes")
    with zipfile.ZipFile(archive) as source_zip:
        if source_zip.namelist() != ["all_train.json"]:
            raise ValueError("unexpected source archive members")
        data = json.loads(source_zip.read("all_train.json"))
    result = summarize(data, json.loads(cut_path.read_text(encoding="utf-8")))
    args.output.mkdir(parents=True, exist_ok=False)
    records_path = args.output / "records.jsonl"
    with records_path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in result.pop("records"):
            stream.write(json.dumps(record, allow_nan=False, sort_keys=True) + "\n")
    report = {
        "status": "TRAIN_ONLY_AV_CORRESPONDENCE_FEASIBILITY",
        "source_revision": SOURCE_REVISION,
        "source_sha256": pins,
        "runner_sha256": sha256_file(Path(__file__)),
        "python_version": platform.python_version(),
        **result,
        "records_sha256": sha256_file(records_path),
        "validation_opened": False,
        "test_opened": False,
        "media_downloaded": False,
        "model_loaded": False,
        "optimizer_updates": 0,
        "gpu_used": False,
        "task_status": "candidate feasibility only; not training-admitted",
        "candidate_rule": (
            "known single parent, is_visible=1, sound interval ends before any annotated cut, "
            "not Muted, and parent is among at least two visible tracks at the "
            "label-blind frame nearest the interval midpoint"
        ),
        "frame_selection_status": (
            "choose among annotated in-interval frames with at least two visible tracks by "
            "distance to interval midpoint, ties to earlier frame; held if source is absent"
        ),
        "option_scope": "supplied ground-truth object tracks; not autonomous object detection",
        "split_scope": "official TRAIN only; video identity is the grouping unit",
        "participant_disjointness": "UNKNOWN; participant IDs are absent from inspected metadata",
    }
    output_path = args.output / "result.json"
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "source_sha256"}, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
