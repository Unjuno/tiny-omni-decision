"""Input-only all-candidate geometric association and separate posthoc accounting."""

import numpy as np


def seed_box_roles(mask_boxes, seed_xy, source_ids, source_boxes):
    """Posthoc mask-IoU>=.5 plus own-point-in-box proxy, never inference labels.

    Missing source ranges must be counted by the caller, not extrapolated. A
    unique available bbox match does not certify visibility or semantic identity.
    Overlap is -2; no available match is -1. All masks remain in input order.
    """
    boxes, xy, ids, targets = map(np.asarray, (mask_boxes, seed_xy, source_ids, source_boxes))
    if (
        boxes.ndim != 2 or boxes.shape[1] != 4 or len(boxes) == 0
        or targets.ndim != 2 or targets.shape[1] != 4
        or ids.shape != (len(targets),) or ids.dtype.kind not in 'iu'
        or (ids < 0).any() or len(np.unique(ids)) != len(ids)
        or xy.shape != (len(boxes), 2) or xy.dtype.kind not in 'fi'
        or not np.isfinite(xy).all()
    ):
        raise ValueError('aligned finite boxes/own seeds and unique integer source IDs required')
    for value in (boxes, targets):
        if (value.dtype.kind not in 'fi' or not np.isfinite(value).all()
                or (value < 0).any() or (value > 1).any()
                or (value[:, :2] >= value[:, 2:]).any()):
            raise ValueError('positive normalized source/mask boxes required')
    labels = np.full(len(boxes), -1, np.int64)
    if not len(targets):
        return labels
    boxes, targets = boxes.astype(np.float64), targets.astype(np.float64)
    overlap = np.maximum(np.minimum(boxes[:, None, 2:], targets[None, :, 2:])
                         - np.maximum(boxes[:, None, :2], targets[None, :, :2]), 0).prod(-1)
    union = ((boxes[:, 2:] - boxes[:, :2]).prod(-1)[:, None]
             + (targets[:, 2:] - targets[:, :2]).prod(-1)[None] - overlap)
    inside = ((xy[:, None] >= targets[None, :, :2])
              & (xy[:, None] <= targets[None, :, 2:])).all(-1)
    matches = (overlap / union >= .5) & inside
    counts = matches.sum(-1)
    labels[counts == 1] = ids[matches.argmax(-1)[counts == 1]]
    labels[counts > 1] = -2
    return labels


def nearest_anchors(query_xy, anchor_xy):
    """Route every query/time to all anchors; first input index wins exact ties.

    No labels, scores, confidence gate, clipping or semantic container filter.
    Distances and margins are geometric measurements, not probabilities. Repeated
    anchors/queries stay distinct. Coordinates share the same image-space raster.
    """
    q, a = np.asarray(query_xy), np.asarray(anchor_xy)
    if (
        q.ndim != 3 or a.ndim != 3 or min(q.shape) == 0 or min(a.shape) == 0
        or q.shape[-1] != 2 or a.shape[-1] != 2 or q.shape[1] != a.shape[1]
        or q.dtype.kind != 'f' or a.dtype.kind != 'f'
        or not np.isfinite(q).all() or not np.isfinite(a).all()
    ):
        raise ValueError('aligned finite floating [queries,time,xy] and anchors required')
    with np.errstate(over='ignore', invalid='ignore'):
        delta = q.astype(np.float64)[:, :, None] - a.astype(np.float64).transpose(1, 0, 2)
        distances = np.hypot(delta[..., 0], delta[..., 1])
    if not np.isfinite(distances).all():
        raise ValueError('geometric distances exceed finite float64')
    indices = distances.argmin(-1).astype(np.int64)
    minimum = np.take_along_axis(distances, indices[..., None], -1)[..., 0]
    second = np.partition(distances, 1, axis=-1)[..., 1] if len(a) > 1 else minimum
    return dict(indices=indices, distances=minimum, margins=second - minimum,
                ties=(distances == minimum[..., None]).sum(-1, dtype=np.int64))


def future_relation_summary(expected_ids, selected_source_labels,
                            observation_frames, sample_frames):
    """Posthoc geometry; exclude seed/earlier times, retain all future unknowns.

    Labels -1 unknown and -2 overlapping/ambiguous remain in the denominator.
    Expected identities must be known; caller reports seed-role coverage separately.
    Repeated queries are not independent QA observations or distinct objects.
    """
    expected, labels, born, times = map(np.asarray,
        (expected_ids, selected_source_labels, observation_frames, sample_frames))
    if (
        expected.ndim != 1 or expected.dtype.kind not in 'iu' or (expected < 0).any()
        or born.shape != expected.shape or born.dtype.kind not in 'iu' or (born < 0).any()
        or times.ndim != 1 or len(times) == 0 or times.dtype.kind not in 'iu'
        or (times < 0).any() or (times[1:] <= times[:-1]).any()
        or labels.shape != (len(expected), len(times)) or labels.dtype.kind not in 'iu'
        or (labels < -2).any()
    ):
        raise ValueError('known IDs, aligned integer labels and ordered original frames required')
    selected = labels[times[None] > born[:, None]]
    wanted = np.broadcast_to(expected[:, None], labels.shape)[times[None] > born[:, None]]
    hits = int((selected == wanted).sum())
    return dict(queries=len(expected), future_denominator=int(selected.size), hits=hits,
                geometric_relation_rate=hits / selected.size if selected.size else None,
                unknown=int((selected == -1).sum()), ambiguous=int((selected == -2).sum()),
                resolved=int((selected >= 0).sum()),
                known_wrong=int(((selected >= 0) & (selected != wanted)).sum()))
