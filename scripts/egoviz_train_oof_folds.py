"""Fixed session-group folds for training-only out-of-fold prediction."""

import numpy as np


def training_folds(groups: np.ndarray):
    groups = np.asarray(groups)
    unique = np.unique(groups)
    if len(unique) != 24:
        raise ValueError("expected 24 training session groups")
    partitions = np.array_split(np.random.default_rng(17).permutation(unique), 5)
    result = []
    for index in range(5):
        heldout = partitions[index]
        inner = partitions[(index + 1) % 5]
        heldout_mask = np.isin(groups, heldout)
        inner_mask = np.isin(groups, inner)
        fit_mask = ~(heldout_mask | inner_mask)
        if np.any(heldout_mask & inner_mask) or not fit_mask.any():
            raise ValueError("overlapping or empty split")
        result.append((fit_mask, inner_mask, heldout_mask))
    return result
