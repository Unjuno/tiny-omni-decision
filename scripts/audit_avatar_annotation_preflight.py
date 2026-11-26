"""Audit extracted AVATAR train/validation annotations without reading audit labels."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

CLIP_ID = re.compile(r"([A-Za-z0-9_-]{11})_(\d+)\Z")
TASKS = {"Single-Sound", "Mixed-Sound", "Multi-Entity", "Off-Screen"}
EXPECTED_JSON_COUNTS = {"train": 19_159, "validation": 2_523}
EXPECTED_CLIP_COUNTS = {"train": 3_952, "validation": 520}
EXPECTED_SOURCE_GROUP_COUNTS = {"train": 3_286, "validation": 411}


def audit_annotations(root: Path, *, check_pins: bool = True) -> dict:
    root = Path(root)
    result: dict[str, object] = {
        "status": "PASS_TRAIN_VALIDATION_ANNOTATION_PREFLIGHT_AUDIT_UNREAD",
        "audit_files_opened": False,
        "splits": {},
    }
    for split in ("train", "validation"):
        split_root = root / split
        files = sorted(split_root.rglob("*.json"))
        if not split_root.is_dir():
            raise ValueError(f"missing {split} annotation directory")
        if check_pins and len(files) != EXPECTED_JSON_COUNTS[split]:
            raise ValueError(f"{split} annotation JSON count differs from the pinned extraction")

        clips: dict[str, dict[str, object]] = defaultdict(
            lambda: {"frame_count": 0, "single_sound_frames": 0, "single_sound_categories": set()}
        )
        groups: set[str] = set()
        seen_frames: set[tuple[str, int]] = set()
        tasks: Counter[str] = Counter()
        task_frames: Counter[str] = Counter()
        annotation_counts: Counter[int] = Counter()
        bbox_by_task: Counter[str] = Counter()
        missing_bbox_by_task: Counter[str] = Counter()
        out_of_frame_bbox = 0

        for path in files:
            clip_id = path.parent.name
            match = CLIP_ID.fullmatch(clip_id)
            if match is None:
                raise ValueError(f"invalid AVATAR clip folder: {clip_id}")
            source_id, clip_offset = match.groups()
            groups.add(source_id)
            frame = json.loads(path.read_text(encoding="utf-8"))
            expected_video_id = f"{source_id}_{int(clip_offset)}"
            if frame.get("video_id") != expected_video_id:
                raise ValueError(f"video_id does not match normalized clip identity: {clip_id}")
            frame_number = frame.get("frame_number")
            if not isinstance(frame_number, int) or frame_number < 0:
                raise ValueError(f"invalid frame number in {path}")
            identity = (frame["video_id"], frame_number)
            if identity in seen_frames:
                raise ValueError(f"duplicate video/frame annotation: {identity}")
            seen_frames.add(identity)

            width, height = frame.get("original_width"), frame.get("original_height")
            if (
                not isinstance(width, int)
                or not isinstance(height, int)
                or width <= 0
                or height <= 0
            ):
                raise ValueError(f"invalid frame dimensions in {path}")
            annotations = frame.get("annotations")
            if not isinstance(annotations, list) or not annotations:
                raise ValueError(f"missing annotation list in {path}")
            annotation_counts[len(annotations)] += 1
            frame_tasks: set[str] = set()
            single_sound_categories: set[str] = set()
            clip = clips[clip_id]
            clip["frame_count"] = int(clip["frame_count"]) + 1

            for annotation in annotations:
                task = annotation.get("task")
                category = annotation.get("audio_visual_category")
                if task not in TASKS or not isinstance(category, str) or not category:
                    raise ValueError(f"unknown/missing task or category in {path}")
                tasks[task] += 1
                frame_tasks.add(task)
                bbox = annotation.get("bbox")
                if task == "Off-Screen":
                    if bbox is not None:
                        raise ValueError(f"Off-Screen annotation unexpectedly has a bbox in {path}")
                    missing_bbox_by_task[task] += 1
                    continue

                if (
                    not isinstance(bbox, list)
                    or len(bbox) != 4
                    or not all(
                        isinstance(value, (int, float)) and math.isfinite(value)
                        for value in bbox
                    )
                    or bbox[2] <= 0
                    or bbox[3] <= 0
                ):
                    raise ValueError(f"invalid visible-source bbox in {path}")
                x, y, box_width, box_height = bbox
                if x < -1 or y < -1 or x + box_width > width + 1 or y + box_height > height + 1:
                    out_of_frame_bbox += 1
                bbox_by_task[task] += 1
                if task == "Single-Sound":
                    single_sound_categories.add(category)

            if single_sound_categories:
                clip["single_sound_frames"] = int(clip["single_sound_frames"]) + 1
                clip["single_sound_categories"].update(single_sound_categories)  # type: ignore[union-attr]

            for task in frame_tasks:
                task_frames[task] += 1

        if out_of_frame_bbox:
            raise ValueError(f"{out_of_frame_bbox} visible-source boxes extend outside the frame")
        eligible_clips = {
            clip_id
            for clip_id, item in clips.items()
            if int(item["single_sound_frames"]) >= 4
            and len(item["single_sound_categories"]) >= 2  # type: ignore[arg-type]
        }
        eligible_groups = {
            CLIP_ID.fullmatch(clip_id).group(1)  # type: ignore[union-attr]
            for clip_id in eligible_clips
        }
        summary = {
            "annotation_json_records": len(files),
            "unique_clip_folders": len(clips),
            "unique_source_youtube_groups": len(groups),
            "frame_records_by_annotation_count": dict(sorted(annotation_counts.items())),
            "task_instance_counts": dict(sorted(tasks.items())),
            "task_frame_record_counts": dict(sorted(task_frames.items())),
            "positive_bbox_counts_by_task": dict(sorted(bbox_by_task.items())),
            "missing_bbox_counts_by_task": dict(sorted(missing_bbox_by_task.items())),
            "out_of_frame_positive_bbox_count": out_of_frame_bbox,
            "potential_temporal_multicategory_clips": len(eligible_clips),
            "potential_temporal_multicategory_source_groups": len(eligible_groups),
        }
        if check_pins and (
            summary["unique_clip_folders"] != EXPECTED_CLIP_COUNTS[split]
            or summary["unique_source_youtube_groups"] != EXPECTED_SOURCE_GROUP_COUNTS[split]
        ):
            raise ValueError(f"{split} clip/source-group counts differ from the pinned split")
        result["splits"][split] = summary  # type: ignore[index]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = audit_annotations(args.annotations_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
