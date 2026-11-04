"""Deterministic per-image augmentation without consuming shuffle RNG."""

import numpy as np


def mirror_mask(count, epoch, seed=17):
    if count < 1 or epoch < 1:
        raise ValueError("positive count and epoch required")
    return np.random.default_rng(np.random.SeedSequence([seed, epoch])).random(count) < 0.5
