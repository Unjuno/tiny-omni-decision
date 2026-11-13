"""Annotation-free soft-mask geometry in normalized image coordinates."""

import numpy as np


def mask_geometry(masks):
    masks = np.asarray(masks, dtype=np.float64)
    if (masks.ndim < 3 or min(masks.shape[-2:]) < 2
            or not np.isfinite(masks).all() or np.any(masks < 0)
            or np.any(masks > 1)):
        raise ValueError("finite nonnegative probability masks required")
    mass = masks.sum(axis=(-2, -1))
    if np.any(mass <= 0):
        raise ValueError("positive mask mass required")
    h, w = masks.shape[-2:]
    y, x = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w), indexing="ij")
    cx = (masks * x).sum(axis=(-2, -1)) / mass
    cy = (masks * y).sum(axis=(-2, -1)) / mass
    sx = np.sqrt((masks * (x - cx[..., None, None]) ** 2).sum(axis=(-2, -1)) / mass)
    sy = np.sqrt((masks * (y - cy[..., None, None]) ** 2).sum(axis=(-2, -1)) / mass)
    return np.stack([cx, cy, sx, sy, mass / (h * w)], axis=-1).astype(np.float32)
