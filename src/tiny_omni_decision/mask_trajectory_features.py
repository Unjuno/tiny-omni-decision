"""Pool frozen point histories by model-produced initial masks, without labels."""

import numpy as np


def mask_query_points(proposal_masks, *, width=256, height=256):
    """Return one foreground pixel-center query per mask and nonempty flags.

    Choose the foreground pixel nearest the mask's center of mass; row-major
    ties are deterministic. Concave masks never seed a point in a hole. Empty
    masks retain [0,0] placeholders with False flags, never valid tracker queries.
    This selects from model-produced masks, not GT identity/location or scores.
    """
    masks = np.asarray(proposal_masks)
    if masks.ndim != 3 or min(masks.shape) < 1 or masks.dtype != np.bool_:
        raise ValueError('nonempty boolean [proposals,height,width] masks required')
    if not np.isfinite([width, height]).all() or min(width, height) <= 0:
        raise ValueError('positive finite tracker raster dimensions required')
    points = np.zeros((len(masks), 2), dtype=np.float64)
    valid = masks.any(axis=(1, 2))
    for i in np.flatnonzero(valid):
        ys, xs = np.nonzero(masks[i])
        count = len(xs)
        # Compare n^2 * distance to COM exactly, including row-major ties.
        bound = 2 * (count * max(masks.shape[1] - 1, masks.shape[2] - 1)) ** 2
        dtype = np.int64 if bound <= np.iinfo(np.int64).max else object
        pixels = np.column_stack((xs, ys)).astype(dtype)
        centered = count * pixels - pixels.sum(axis=0)
        chosen = pixels[np.square(centered).sum(axis=1).argmin()].astype(np.float64)
        points[i] = (chosen + .5) / [masks.shape[2], masks.shape[1]] * [width, height]
    return points, valid


def proposal_point_membership(query_xy, proposal_masks, *, width=256, height=256):
    """Return [proposals, points] membership at the original query positions.

    Queries use (x,y) in the tracker raster. Masks span that same normalized
    frame, possibly at another resolution. Outside points are false, never
    clipped into a boundary object. Overlapping masks retain shared support.
    This initial support is neither semantic identity nor tracking validity.
    """
    raw_xy = np.asarray(query_xy)
    if raw_xy.dtype.kind not in 'biuf':
        raise ValueError('real numeric queries required')
    with np.errstate(over='ignore', invalid='ignore'):
        xy = raw_xy.astype(np.float64)
    masks = np.asarray(proposal_masks)
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) == 0 or not np.isfinite(xy).all():
        raise ValueError('nonempty finite [points,xy] queries required')
    if masks.ndim != 3 or min(masks.shape) < 1 or masks.dtype != np.bool_:
        raise ValueError('nonempty boolean [proposals,height,width] masks required')
    if not np.isfinite([width, height]).all() or min(width, height) <= 0:
        raise ValueError('positive finite tracker raster dimensions required')
    inside = ((xy >= 0) & (xy < [width, height])).all(axis=1)
    members = np.zeros((len(masks), len(xy)), dtype=bool)
    pixels = np.zeros((int(inside.sum()), 2), dtype=np.int64)
    mask_size_xy = (masks.shape[2], masks.shape[1])
    for axis, (raster, mask_size) in enumerate(zip((width, height), mask_size_xy, strict=True)):
        values = xy[inside, axis]
        if raster == mask_size:
            pixels[:, axis] = np.floor(values).astype(np.int64)
        else:
            # Exact binary-rational floor avoids rounding just below a boundary
            # into the next cell; only in-frame points are converted.
            raster_n, raster_d = float(raster).as_integer_ratio()
            pixels[:, axis] = [
                (n * mask_size * raster_d) // (d * raster_n)
                for n, d in (float(v).as_integer_ratio() for v in values)
            ]
    members[:, inside] = masks[:, pixels[:, 1], pixels[:, 0]]
    return members


def mask_trajectory_features(point_features, query_xy, proposal_masks, *, width=256, height=256):
    """Return (tokens, supported, counts), retaining every proposal in input order.

    Average points within each initial mask, keeping every temporal/appearance
    column in its original position. Unsupported masks return zero tokens with
    supported=False and counts=0; callers must keep these flags. No future mask,
    ground-truth selector, confidence filter, TopK or exclusive assignment is
    applied. The function adds no learned parameters or new encoder inference.
    """
    raw_features = np.asarray(point_features)
    if raw_features.dtype.kind not in 'biuf':
        raise ValueError('real numeric point features required')
    with np.errstate(over='ignore', invalid='ignore'):
        features = raw_features.astype(np.float32)
    membership = proposal_point_membership(query_xy, proposal_masks, width=width, height=height)
    if (
        features.ndim != 2 or features.shape[0] != membership.shape[1]
        or features.shape[1] == 0 or not np.isfinite(features).all()
    ):
        raise ValueError('aligned nonempty finite [points,features] required')
    counts = membership.sum(axis=1, dtype=np.int64)
    supported = counts > 0
    # Accumulate in float64 so a finite float32 mean cannot overflow its sum.
    summed = membership.astype(np.float64) @ features.astype(np.float64)
    tokens = np.zeros(summed.shape, dtype=np.float32)
    tokens[supported] = summed[supported] / counts[supported, None]
    return tokens, supported, counts
