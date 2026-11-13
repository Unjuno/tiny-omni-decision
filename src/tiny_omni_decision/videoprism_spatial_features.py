"""Spatial intermediate readout contract, distinct from legacy pooled caches."""

import numpy as np


def spatial_cells(tokens):
    x = np.asarray(tokens)
    if x.ndim != 3 or x.shape[1:] != (256, 768) or not np.isfinite(x).all():
        raise ValueError("finite [batch,256,768] spatial tokens required")
    cells = x.reshape(len(x), 4, 4, 4, 4, 768).mean(axis=(2, 4)).reshape(len(x), 16, 768)
    return np.concatenate((x.mean(1, keepdims=True), cells), axis=1)


def readout_features(cells, condition):
    x = np.asarray(cells)
    if x.ndim != 3 or x.shape[1:] != (17, 768) or not np.isfinite(x).all():
        raise ValueError("finite [batch,17,768] features required")
    if condition not in {"spatial", "global"}:
        raise ValueError("unknown representation")
    parts = x[:, 1:]
    if condition == "global":
        parts = np.repeat(parts.mean(1, keepdims=True), 16, axis=1)
    return np.concatenate((x[:, 0], parts.reshape(len(x), -1) / 4), axis=1)
