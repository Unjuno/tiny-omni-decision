"""CPU-testable contracts for the fixed DINOv2 image probe."""

import numpy as np


def cls_patch_mean(tokens):
    x = np.asarray(tokens)
    if x.ndim != 3 or x.shape[1:] != (257, 384) or not np.isfinite(x).all():
        raise ValueError("expected finite [batch,257,384] tokens")
    return np.concatenate([x[:, 0], x[:, 1:].mean(1)], axis=1)


def require_image_splits(train, validation):
    for field in ("id", "sha256"):
        a, b = [r[field] for r in train], [r[field] for r in validation]
        if not a or not b or len(a) != len(set(a)) or len(b) != len(set(b)) or set(a) & set(b):
            raise ValueError(f"duplicate, empty or cross-split image {field}")
