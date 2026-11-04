"""Annotation-independent, ordered trajectory tokens for frozen-feature probes."""

import numpy as np


def trajectory_features(tracks, occlusion, expected_dist, *, width=256, height=256, samples=32):
    """Retain every query; sample time uniformly and append sigmoid logits.

    Coordinates use TAPIR raster width/height units and are not clipped: points
    outside the image carry information. Uncertainty channels are model outputs,
    not calibrated Decision probabilities. No labels or object boxes are inputs.
    """
    tracks = np.asarray(tracks, dtype=np.float32)
    occlusion = np.asarray(occlusion, dtype=np.float32)
    expected_dist = np.asarray(expected_dist, dtype=np.float32)
    if tracks.ndim != 3 or tracks.shape[-1] != 2 or min(tracks.shape[:2]) < 1:
        raise ValueError("nonempty [points,time,xy] tracks required")
    if occlusion.shape != tracks.shape[:2] or expected_dist.shape != tracks.shape[:2]:
        raise ValueError("aligned uncertainty arrays required")
    if not all(np.isfinite(a).all() for a in (tracks, occlusion, expected_dist)):
        raise ValueError("nonfinite tracker output")
    if not isinstance(samples, int) or samples < 2:
        raise ValueError("at least two temporal samples required")
    if not np.isfinite([width, height]).all() or min(width, height) <= 0:
        raise ValueError("positive finite raster dimensions required")
    indices = np.rint(np.linspace(0, tracks.shape[1] - 1, samples)).astype(np.int64)
    xy = tracks[:, indices] / np.array([width, height], dtype=np.float32)

    def sigmoid(values):
        # Stable for arbitrarily large finite logits, without overflow.
        e = np.exp(-np.abs(values))
        return np.where(values >= 0, 1 / (1 + e), e / (1 + e))

    features = np.concatenate(
        [xy, sigmoid(occlusion[:, indices, None]), sigmoid(expected_dist[:, indices, None])],
        axis=-1,
    )
    return features.reshape(len(tracks), samples * 4), indices
