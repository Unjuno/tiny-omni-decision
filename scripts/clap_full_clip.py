"""Full sample coverage and label-independent duration-weighted CLAP pooling."""

import numpy as np


def window_spans(length, maximum):
    if (
        not isinstance(length, int)
        or not isinstance(maximum, int)
        or isinstance(length, bool)
        or isinstance(maximum, bool)
        or length < 1
        or maximum < 1
    ):
        raise ValueError("positive integer sample counts required")
    return [(start, min(start + maximum, length)) for start in range(0, length, maximum)]


def duration_weighted_feature(features, sample_counts):
    values = np.asarray(features, dtype=np.float64)
    counts = np.asarray(sample_counts, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[1] < 1
        or counts.shape != (len(values),)
        or not np.isfinite(values).all()
        or not np.isfinite(counts).all()
        or not len(values)
        or (counts <= 0).any()
    ):
        raise ValueError("finite window vectors and positive aligned durations required")
    pooled = (values * (counts / counts.sum())[:, None]).sum(axis=0)
    norm = np.linalg.norm(pooled)
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("pooled feature has undefined direction")
    return (pooled / norm).astype(np.float32)
