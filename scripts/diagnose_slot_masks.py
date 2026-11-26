"""Oracle-initialized slot-mask diagnostics; these are not Decision metrics."""

import numpy as np


def box_mask(box, height, width):
    box = np.asarray(box, dtype=float)
    if box.shape != (4,) or not np.isfinite(box).all():
        raise ValueError("invalid box")
    if not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1):
        raise ValueError("invalid normalized box bounds")
    x = (np.arange(width) + 0.5) / width
    y = (np.arange(height) + 0.5) / height
    return ((x >= box[0]) & (x < box[2]))[None, :] & ((y >= box[1]) & (y < box[3]))[:, None]


def iou(a, b):
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def compare_masks(masks, frame_ids, boxes):
    masks = np.asarray(masks)
    if masks.ndim != 4 or not np.isfinite(masks).all():
        raise ValueError("expected finite [time,slot,height,width]")
    if len(masks) != len(frame_ids) or len(set(frame_ids)) != len(frame_ids):
        raise ValueError("unaligned or duplicate frame IDs")
    if 0 not in boxes or 0 not in frame_ids:
        return dict(status="NO_FRAME0_BOX")
    height, width = masks.shape[-2:]
    hard = masks.argmax(axis=1)
    initial = box_mask(boxes[0], height, width)
    first = frame_ids.index(0)
    scores = [iou(hard[first] == slot, initial) for slot in range(masks.shape[1])]
    selected = int(np.argmax(scores))
    points = []
    for index, frame in enumerate(frame_ids):
        if frame not in boxes:
            continue
        box = box_mask(boxes[frame], height, width)
        scores = [iou(hard[index] == slot, box) for slot in range(masks.shape[1])]
        points.append(
            dict(
                frame=int(frame),
                fixed_iou=scores[selected],
                oracle_iou=max(scores),
                oracle_slot=int(np.argmax(scores)),
                static_iou=iou(initial, box),
                selected_area=float((hard[index] == selected).mean()),
                largest_slot_area=float(
                    max((hard[index] == s).mean() for s in range(masks.shape[1]))
                ),
            )
        )
    return dict(status="ORACLE_BOX_DIAGNOSTIC", selected_slot=selected, points=points)
