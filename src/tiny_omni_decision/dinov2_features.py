"""CPU-testable contracts for the fixed DINOv2 image probe."""

import numpy as np


def cls_patch_mean(tokens):
    x = np.asarray(tokens)
    if x.ndim != 3 or x.shape[1:] != (257, 384) or not np.isfinite(x).all():
        raise ValueError("expected finite [batch,257,384] tokens")
    return np.concatenate([x[:, 0], x[:, 1:].mean(1)], axis=1)


def spatial_cells(tokens):
    x = np.asarray(tokens)
    cls_patch_mean(x)  # Shared finite/shape contract.
    grid = x[:, 1:].reshape(len(x), 4, 4, 4, 4, 384)
    cells = grid.mean(axis=(2, 4)).reshape(len(x), 16, 384)
    # Align only the rounding offset, identically for every spatial cell.
    # The direct FP32 mean is the frozen baseline's actual representation.
    reference = x[:, 1:].mean(1, keepdims=True).astype(np.float64)
    precise = cells.astype(np.float64)
    cells = (precise + reference - precise.mean(1, keepdims=True)).astype(x.dtype)
    return np.concatenate([x[:, :1], cells], axis=1)


def readout_features(cells, condition):
    x = np.asarray(cells)
    if x.ndim != 3 or x.shape[1:] != (17, 384) or not np.isfinite(x).all():
        raise ValueError("invalid spatial feature cache")
    if condition not in {"global", "spatial"}:
        raise ValueError("unknown representation")
    patch = x[:, 1:]
    if condition == "global":
        patch = np.repeat(patch.mean(1, keepdims=True), 16, axis=1)
    return np.concatenate([x[:, 0], patch.reshape(len(x), -1) / 4], axis=1)


def require_image_splits(train, validation):
    for field in ("id", "sha256"):
        a, b = [r[field] for r in train], [r[field] for r in validation]
        if not a or not b or len(a) != len(set(a)) or len(b) != len(set(b)) or set(a) & set(b):
            raise ValueError(f"duplicate, empty or cross-split image {field}")
