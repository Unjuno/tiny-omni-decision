"""Development-only paired cluster resampling; caller must align sample IDs."""

import numpy as np


def paired_cluster_interval(left, right, groups):
    """Fixed seed17, 10,000 cluster draws and percentile/linear95% interval.

    Estimates sample-weighted left-minus-right mean; all observations belonging
    to a sampled cluster move together. Does not correct checkpoint selection
    bias or make a small number of participants representative of a population.
    """
    a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    group_array = np.asarray(groups)
    if (a.ndim != 1 or not a.size or a.shape != b.shape or a.shape != group_array.shape
            or not np.isfinite(a).all() or not np.isfinite(b).all()):
        raise ValueError("finite aligned nonempty pairs required")
    unique = sorted(set(groups))
    if len(unique) < 2:
        raise ValueError("at least two clusters required")
    differences = a - b
    sums = np.array([differences[group_array == group].sum() for group in unique])
    counts = np.array([(group_array == group).sum() for group in unique])
    draws = np.random.default_rng(17).integers(0, len(unique), size=(10000, len(unique)))
    estimates = sums[draws].sum(1) / counts[draws].sum(1)
    return {"mean_difference": float(differences.mean()),
            "interval_95": np.quantile(estimates, [.025, .975], method="linear").tolist(),
            "clusters": len(unique), "samples": len(a), "replicates": 10000, "seed": 17,
            "method": "paired cluster resampling, percentile95, linear quantiles"}
