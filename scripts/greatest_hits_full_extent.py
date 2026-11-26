"""Bind real event frames to the existing full-extent policy and event clock."""

import numpy as np

from scripts.greatest_hits_features import event_positions
from scripts.greatest_hits_inputs import select_video
from scripts.native_full_extent import POLICY, preprocess_full_extent


def prepare_review_video(rgb, pts, start, end, tf):
    stamps = np.asarray(pts)
    if (not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 4
            or rgb.shape[1:] != (256, 456, 3) or not 8 <= len(rgb) <= 16
            or stamps.shape != (len(rgb),)
            or not ((stamps >= start) & (stamps < end)).all()):
        raise ValueError('complete original RGB/PTS event window required')
    indices = select_video(stamps, start, end)
    selected = stamps[indices]
    positions = event_positions(selected, start, end)['video']
    prepared, geometry = preprocess_full_extent(rgb[indices], tf)
    return prepared, positions, dict(policy=POLICY, geometry=geometry,
        selected_indices=indices, selected_pts=selected.tolist(),
        spatial_coordinates='encoder canvas including padding')
