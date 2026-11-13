"""Posthoc TRAIN bbox proxies; never seeds or semantic Decision accuracy."""

import numpy as np


def _assignment(pred, source, available):
    if not available.all():
        # Unobserved group geometry could overlap this prediction; do not infer
        # uniqueness only among whichever source boxes happen to be available.
        return np.full(len(pred), -1, np.int64)
    lo = np.maximum(pred[:, None, :2], source[None, :, :2])
    hi = np.minimum(pred[:, None, 2:], source[None, :, 2:])
    intersection = np.maximum(hi - lo, 0).prod(-1)
    area_pred = np.maximum(pred[:, 2:] - pred[:, :2], 0).prod(-1)
    area_source = np.maximum(source[:, 2:] - source[:, :2], 0).prod(-1)
    union = area_pred[:, None] + area_source[None, :] - intersection
    iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
    hits = (iou >= 0.5) & available[None, :]
    counts = hits.sum(-1)
    return np.where(counts == 1, hits.argmax(-1), np.where(counts == 0, -1, -2))


def evaluate_group_proxy(actual, repeat, source, available, source_ids):
    """All source groups retained; unknown future observations stay in denominators."""
    if (
        actual.ndim != 3
        or actual.shape[-1] != 4
        or actual.shape[1] < 2
        or actual.shape[0] < 1
        or repeat.shape != actual.shape
        or source.ndim != 3
        or source.shape[-1] != 4
        or source.shape[0] != actual.shape[1]
        or source.shape[1] < 1
        or available.shape != source.shape[:2]
        or available.dtype != bool
        or source_ids.shape != (source.shape[1],)
        or source_ids.dtype != np.int64
        or len(np.unique(source_ids)) != len(source_ids)
        or actual.dtype != np.float64
        or repeat.dtype != np.float64
        or source.dtype != np.float64
        or not np.isfinite(actual).all()
        or not np.isfinite(repeat).all()
        or not np.isfinite(source[available]).all()
        or ((source[available, 2:] - source[available, :2]) <= 0).any()
        or actual[:, 0].tobytes() != repeat[:, 0].tobytes()
    ):
        raise ValueError("complete fixed source/prediction geometry and identical frame0 required")
    for b in (actual, repeat):
        bad_area = ((b[..., 2:] - b[..., :2]) <= 0).any(-1)
        if (b < 0).any() or (b > 1).any() or (bad_area & (b != 0).any(-1)).any():
            raise ValueError("native normalized raster boxes or explicit empty0 required")
    # Missing amodal source boxes remain missing, never interpolated further here.
    safe_source = np.where(available[..., None], source, 0.0)
    initial = _assignment(actual[:, 0], safe_source[0], available[0])
    eligible = initial >= 0
    group_count = len(source_ids)
    future = actual.shape[1] - 1
    result = dict(
        source_groups=group_count,
        source_ids=source_ids.tolist(),
        initial_source_groups_supported=len(np.unique(initial[eligible])),
        initial_source_group_supported_ids=source_ids[np.unique(initial[eligible])].tolist(),
        anchor_counts=dict(
            total=len(initial),
            eligible=int(eligible.sum()),
            unknown=int((initial == -1).sum()),
            ambiguous=int((initial == -2).sum()),
        ),
        anchors=[int(source_ids[a]) if a >= 0 else None for a in initial],
        anchor_status=initial.tolist(),
        source_future_available=int(available[1:].sum()),
        source_future_unavailable=int((~available[1:]).sum()),
        arms={},
    )
    for name, boxes in [("actual", actual), ("initial_repeat", repeat)]:
        categories = {
            k: 0
            for k in [
                "same_unique_group",
                "different_unique_group",
                "ambiguous",
                "no_group_match",
                "empty_mask",
                "target_source_unavailable",
                "incomplete_group_source",
            ]
        }
        covered = np.zeros((future, group_count), bool)
        for t in range(1, future + 1):
            assignments = _assignment(boxes[:, t], safe_source[t], available[t])
            for i in np.flatnonzero(eligible):
                anchor = initial[i]
                if not available[t, anchor]:
                    key = "target_source_unavailable"
                elif not available[t].all():
                    key = "incomplete_group_source"
                elif not boxes[i, t].any():
                    key = "empty_mask"
                elif assignments[i] == anchor:
                    key = "same_unique_group"
                    covered[t - 1, anchor] = True
                elif assignments[i] >= 0:
                    key = "different_unique_group"
                elif assignments[i] == -2:
                    key = "ambiguous"
                else:
                    key = "no_group_match"
                categories[key] += 1
        denominator = int(eligible.sum()) * future
        assert sum(categories.values()) == denominator
        result["arms"][name] = dict(
            future_categories=categories,
            eligible_seed_future_denominator=denominator,
            eligible_seed_future_rate=categories["same_unique_group"] / denominator
            if denominator
            else None,
            source_group_future_denominator=group_count * future,
            source_group_future_covered=int(covered.sum()),
            source_group_future_rate=float(covered.mean()),
            source_group_future_coverage_by_time=covered.astype(int).tolist(),
        )
    return result
