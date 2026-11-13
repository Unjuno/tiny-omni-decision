"""Input-only descriptor retrieval and separate posthoc geometric labels."""

import numpy as np


def association_summary(initial_labels, candidate_labels, indices):
    """All-query accounting; future unknown matches remain in the denominator."""
    initial, labels, indices = map(np.asarray, (initial_labels, candidate_labels, indices))
    if (
        initial.ndim != 1
        or len(initial) == 0
        or labels.ndim != 2
        or min(labels.shape) == 0
        or labels.shape[1] < 2
        or indices.shape != (len(initial), labels.shape[1])
        or initial.dtype.kind not in "iu"
        or labels.dtype.kind not in "iu"
        or indices.dtype.kind not in "iu"
        or (initial < -2).any()
        or (labels < -2).any()
        or (indices < 0).any()
        or (indices >= len(labels)).any()
    ):
        raise ValueError("aligned integer labels and bounded prediction indices required")
    selected = labels[indices, np.arange(labels.shape[1])[None]]
    known = initial >= 0
    future = selected[known, 1:]
    hits = future == initial[known, None]
    resolved = future >= 0
    return dict(
        all_query_point_times=int(indices.size),
        initial_group_queries=int(known.sum()),
        initial_unknown_queries=int((initial == -1).sum()),
        initial_ambiguous_queries=int((initial == -2).sum()),
        future_denominator=int(future.size),
        future_hits=int(hits.sum()),
        future_same_group_bbox_rate=float(hits.mean()) if future.size else None,
        future_resolved_predictions=int(resolved.sum()),
        future_conditional_resolved_hit_rate=float(hits.sum() / resolved.sum())
        if resolved.any()
        else None,
        future_unknown_predictions=int((future == -1).sum()),
        future_ambiguous_predictions=int((future == -2).sum()),
        time0_hits=int((selected[known, 0] == initial[known]).sum()),
    )


def descriptor_matches(query, candidates):
    """Match every query at every time; deterministic first-index ties, no filters."""
    query, candidates = np.asarray(query), np.asarray(candidates)
    if (
        query.ndim != 2
        or candidates.ndim != 3
        or min(query.shape) == 0
        or min(candidates.shape) == 0
        or query.shape[-1] != candidates.shape[-1]
        or query.dtype.kind != "f"
        or candidates.dtype.kind != "f"
        or not np.isfinite(query).all()
        or not np.isfinite(candidates).all()
    ):
        raise ValueError("nonempty aligned finite floating descriptor arrays required")

    def normalized(value):
        value = value.astype(np.float64)
        scale = np.abs(value).max(-1, keepdims=True)
        bounded = np.divide(value, scale, out=np.zeros_like(value), where=scale > 0)
        norm = np.linalg.norm(bounded, axis=-1, keepdims=True)
        return np.divide(bounded, norm, out=np.zeros_like(value), where=norm > 0), scale[
            ..., 0
        ] == 0

    q, zero_query = normalized(query)
    c, zero_candidate = normalized(candidates)
    scores = np.einsum("qc,ntc->qtn", q, c, optimize=True)
    indices = scores.argmax(-1)
    maximum = np.take_along_axis(scores, indices[..., None], -1)[..., 0]
    second = np.partition(scores, -2, axis=-1)[..., -2] if len(c) > 1 else maximum
    return dict(
        indices=indices,
        scores=maximum,
        margins=maximum - second,
        ties=(scores == maximum[..., None]).sum(-1),
        zero_query=zero_query,
        zero_candidate=zero_candidate,
    )


def group_membership(tracks, group_ids, xy, frame_ids):
    """Posthoc bbox proxy: -1 unknown/outside, -2 overlapping group membership.

    Source boxes are linearly interpolated only inside their annotation ranges.
    Group IDs must never select prediction queries/candidates. This is geometry,
    not visibility or semantic point-tracking truth.
    """
    xy, frame_ids = np.asarray(xy), np.asarray(frame_ids)
    if (
        xy.ndim != 3
        or min(xy.shape) == 0
        or xy.shape[-1] != 2
        or xy.dtype.kind != "f"
        or not np.isfinite(xy).all()
        or frame_ids.shape != (xy.shape[1],)
        or frame_ids.dtype.kind not in "iu"
        or (np.diff(frame_ids.astype(np.float64)) <= 0).any()
        or not group_ids
        or len(set(group_ids)) != len(group_ids)
        or any(not isinstance(i, int) or isinstance(i, bool) or i < 0 for i in group_ids)
    ):
        raise ValueError("aligned coordinates, ordered frames and unique group IDs required")
    all_hits, all_available = [], []
    for identity in group_ids:
        selected = [t for t in tracks if t["id"] == identity]
        if len(selected) != 1:
            raise ValueError("exact source group track required")
        track = selected[0]
        frames, boxes = np.asarray(track["frame_ids"]), np.asarray(track["bounding_boxes"])
        if (
            frames.ndim != 1
            or len(frames) == 0
            or frames.dtype.kind not in "iu"
            or (np.diff(frames.astype(np.float64)) <= 0).any()
            or boxes.shape != (len(frames), 4)
            or boxes.dtype.kind not in "fi"
            or not np.isfinite(boxes).all()
            or (boxes < 0).any()
            or (boxes > 1).any()
            or (boxes[:, :2] >= boxes[:, 2:]).any()
        ):
            raise ValueError("ordered source frames and positive normalized boxes required")
        available = (frame_ids >= frames[0]) & (frame_ids <= frames[-1])
        interpolated = np.stack([np.interp(frame_ids, frames, boxes[:, k]) for k in range(4)], -1)
        hits = ((xy >= interpolated[None, :, :2]) & (xy <= interpolated[None, :, 2:])).all(-1)
        hits &= available[None] & ((xy >= 0) & (xy <= 1)).all(-1)
        all_hits.append(hits)
        all_available.append(available)
    hits = np.stack(all_hits, -1)
    count = hits.sum(-1)
    labels = np.full(count.shape, -1, np.int64)
    identities = np.asarray(group_ids, np.int64)
    labels[count == 1] = identities[hits.argmax(-1)[count == 1]]
    labels[count > 1] = -2
    coverage = np.stack(all_available)
    # A missing group's range cannot rule out another bbox membership.
    labels[:, ~coverage.all(0)] = -1
    return labels, coverage
