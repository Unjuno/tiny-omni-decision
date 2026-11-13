"""Exact index interpretation of the pinned eight-frame VideoPrism output."""

import numpy as np


def token_grid(tokens):
    values = np.asarray(tokens)
    if values.shape != (1, 2048, 768) or values.dtype != np.float32:
        raise ValueError("expected pinned FP32 [1,2048,768] tokens")
    if not np.isfinite(values).all():
        raise ValueError("nonfinite tokens")
    return values.reshape(8, 16, 16, 768)
