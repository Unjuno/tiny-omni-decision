"""Posthoc bbox opportunity only; never an input or semantic target selector."""

import numpy as np


def valid_boxes(value, *, positive=False):
    boxes = np.asarray(value)
    if (
        boxes.ndim != 2
        or len(boxes) == 0
        or boxes.shape[1] != 4
        or boxes.dtype.kind not in "fi"
        or not np.isfinite(boxes).all()
        or (boxes < 0).any()
        or (boxes > 1).any()
        or (boxes[:, :2] > boxes[:, 2:]).any()
        or (positive and (boxes[:, :2] >= boxes[:, 2:]).any())
    ):
        raise ValueError("finite normalized bbox geometry required")
    return boxes.astype(np.float64)


def bbox_opportunity(boxes, nms_indices, target_box):
    boxes = valid_boxes(boxes)
    indices = np.asarray(nms_indices)
    if (
        indices.ndim != 1
        or indices.dtype.kind not in "iu"
        or (indices < 0).any()
        or (indices >= len(boxes)).any()
        or len(np.unique(indices)) != len(indices)
    ):
        raise ValueError("unique bounded NMS indices required")
    if target_box is None:
        return dict(
            source_box_available=False,
            raw_max_iou=None,
            nms_max_iou=None,
            raw_best_index=None,
            nms_best_index=None,
        )
    target = valid_boxes(np.asarray(target_box)[None], positive=True)[0]
    low = np.maximum(boxes[:, :2], target[:2])
    high = np.minimum(boxes[:, 2:], target[2:])
    overlap = np.maximum(high - low, 0).prod(1)
    union = (boxes[:, 2:] - boxes[:, :2]).prod(1) + (target[2:] - target[:2]).prod() - overlap
    values = np.divide(overlap, union, out=np.zeros_like(union), where=union > 0)
    raw_best = int(values.argmax())
    kept_best = int(indices[values[indices].argmax()]) if len(indices) else None
    return dict(
        source_box_available=True,
        raw_max_iou=float(values[raw_best]),
        nms_max_iou=float(values[kept_best]) if kept_best is not None else 0.0,
        raw_best_index=raw_best,
        nms_best_index=kept_best,
    )


def source_box_at_time(track, frame):
    """Range-limited sparse annotation proxy; masked flags do not imply visibility."""
    frames = np.asarray(track["frame_ids"])
    boxes = valid_boxes(track["bounding_boxes"], positive=True)
    if (
        frames.ndim != 1
        or len(frames) != len(boxes)
        or frames.dtype.kind not in "iu"
        or (frames < 0).any()
        or (np.diff(frames.astype(np.float64)) <= 0).any()
        or not isinstance(frame, int)
        or isinstance(frame, bool)
        or frame < 0
    ):
        raise ValueError("nonnegative ordered source frames and integer query time required")
    flags = track.get("is_masked")
    if flags is not None and (len(flags) != len(frames) or any(x not in (0, 1) for x in flags)):
        raise ValueError("aligned source masked flags required")
    if frame < frames[0] or frame > frames[-1]:
        return dict(
            box=None,
            bracket_frames=None,
            source_frame_exact=False,
            nearest_annotation_distance_frames=None,
            source_masked_flags=None,
        )
    right = int(np.searchsorted(frames, frame))
    left = right if frames[right] == frame else right - 1
    box = [float(np.interp(frame, frames, boxes[:, k])) for k in range(4)]
    return dict(
        box=box,
        bracket_frames=[int(frames[left]), int(frames[right])],
        source_frame_exact=left == right,
        nearest_annotation_distance_frames=int(min(frame - frames[left], frames[right] - frame)),
        source_masked_flags=[flags[left], flags[right]] if flags is not None else None,
    )
