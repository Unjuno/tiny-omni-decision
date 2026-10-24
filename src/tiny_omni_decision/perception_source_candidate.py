"""Train-only adapter for audio-event to visual-source candidate matching."""

import hashlib
import math


def _valid_box(box):
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise ValueError("bounding boxes must be normalized xyxy quadruples")
    if any(type(value) not in (float, int) or not math.isfinite(value) for value in box):
        raise ValueError("bounding boxes must contain finite numbers")
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise ValueError("bounding boxes must be valid normalized xyxy coordinates")
    return [float(value) for value in box]


def _sample_frames(frames, limit=4):
    if len(frames) <= limit:
        return frames
    # Evenly spaced indices are deterministic and include both interval ends.
    indices = [round(index * (len(frames) - 1) / (limit - 1)) for index in range(limit)]
    return [frames[index] for index in indices]


def source_candidate_records(video_id, row):
    """Yield candidate-reranking records; annotation labels stay in supervision only.

    `frame_ids` on sound annotations are inclusive interval bounds. Candidate object
    tracks contribute only unmasked boxes inside those bounds. The task requires
    externally supplied audio and frame crops; this function emits references and
    targets, not media or autonomous event detections.
    """
    metadata = row["metadata"]
    if metadata.get("split") != "train" or metadata.get("video_id") != video_id:
        raise ValueError("train-only matching video identity required")
    if type(metadata.get("num_frames")) is not int or metadata["num_frames"] <= 0:
        raise ValueError("positive integer video frame count required")

    objects = row.get("object_tracking", [])
    by_id = {obj["id"]: obj for obj in objects}
    if len(by_id) != len(objects):
        raise ValueError("duplicate object annotation IDs")
    for obj in objects:
        frame_ids = obj.get("frame_ids")
        boxes = obj.get("bounding_boxes")
        masked = obj.get("is_masked")
        if (
            not isinstance(frame_ids, list)
            or not isinstance(boxes, list)
            or len(frame_ids) != len(boxes)
        ):
            raise ValueError("object frame IDs and boxes must align")
        if masked is None:
            masked = [0] * len(frame_ids)
        if not isinstance(masked, list) or len(masked) != len(frame_ids):
            raise ValueError("object mask flags must align with frame IDs")
        for frame_id in frame_ids:
            if type(frame_id) is not int or not 0 <= frame_id < metadata["num_frames"]:
                raise ValueError("object frame ID outside video")
        for box in boxes:
            _valid_box(box)
        if any(type(value) is not int or value not in (0, 1) for value in masked):
            raise ValueError("object mask flags must be binary integers")

    sounds = row.get("sound_localisation", [])
    if len({sound["id"] for sound in sounds}) != len(sounds):
        raise ValueError("duplicate sound annotation IDs")
    for sound in sounds:
        bounds = sound.get("frame_ids")
        timestamps = sound.get("timestamps")
        if (
            not isinstance(bounds, list)
            or len(bounds) != 2
            or any(type(frame) is not int for frame in bounds)
            or bounds[0] < 0
            or bounds[1] < bounds[0]
            or bounds[1] >= metadata["num_frames"]
            or not isinstance(timestamps, list)
            or len(timestamps) != 2
            or any(
                type(value) not in (int, float) or not math.isfinite(value)
                for value in timestamps
            )
            or timestamps[0] < 0
            or timestamps[1] <= timestamps[0]
        ):
            raise ValueError("invalid sound interval bounds")
        if type(sound.get("is_visible")) is not int or sound["is_visible"] not in (0, 1):
            raise ValueError("sound visibility must be a binary integer")
        parents = sound.get("parent_objects")
        if not isinstance(parents, list) or any(type(parent) is not int for parent in parents):
            raise ValueError("parent object IDs must be integer list")

    for sound in sounds:
        sound_id = sound["id"]
        start, end = sound["frame_ids"]
        parent_ids = sound["parent_objects"]
        hold_reasons = []
        if sound["is_visible"] != 1:
            hold_reasons.append("source_not_annotated_visible")
        if not parent_ids or -1 in parent_ids:
            hold_reasons.append("unknown_or_empty_source_annotation")
        if len(set(parent_ids)) != len(parent_ids):
            hold_reasons.append("duplicate_parent_object_id")

        candidates = []
        candidate_object_ids = set()
        for obj in objects:
            frames = []
            for index, frame_id in enumerate(obj["frame_ids"]):
                mask_flags = obj.get("is_masked", [0] * len(obj["frame_ids"]))
                if start <= frame_id <= end and mask_flags[index] == 0:
                    frames.append(
                        {
                            "frame_id": frame_id,
                            "bbox_xyxy_normalized": _valid_box(obj["bounding_boxes"][index]),
                        }
                    )
            if frames:
                opaque_order_key = hashlib.sha256(
                    f"{video_id}:{sound_id}:{obj['id']}".encode()
                ).hexdigest()
                candidate_object_ids.add(obj["id"])
                candidates.append(
                    {
                        "_order_key": opaque_order_key,
                        "_object_id": obj["id"],
                        "frame_boxes": _sample_frames(frames),
                    }
                )

        if not set(parent_ids).issubset(candidate_object_ids):
            hold_reasons.append("source_object_lacks_visible_interval_box")
        if not candidate_object_ids.difference(parent_ids):
            hold_reasons.append("no_non_source_visual_distractor")
        status = "held" if hold_reasons else "candidate"
        record = {
            "record_id": f"{video_id}:sound:{sound_id}",
            "group_id": video_id,
            "split": "train",
            "status": status,
            "hold_reasons": hold_reasons,
        }
        if status == "candidate":
            candidates.sort(key=lambda candidate: candidate["_order_key"])
            object_to_option = {}
            request_candidates = []
            for option_index, candidate in enumerate(candidates):
                option_id = f"option_{option_index}"
                object_to_option[candidate["_object_id"]] = option_id
                request_candidates.append(
                    {
                        "candidate_id": option_id,
                        "frame_boxes": candidate["frame_boxes"],
                    }
                )
            record["request"] = {
                "task": "audio_visual_source_candidate_reranking",
                "instruction": "Select every visual candidate that produced the audio event.",
                "audio_interval_us": list(sound["timestamps"]),
                "candidates": request_candidates,
            }
            record["media_reference"] = {
                "video_id": video_id,
                "sound_annotation_id": sound_id,
            }
            record["supervision"] = {
                "target_candidate_ids": sorted(object_to_option[parent] for parent in parent_ids)
            }
        yield record
