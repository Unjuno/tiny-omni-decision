"""Exact query cache reuse and missing-vector schedule, without model dependencies."""

import numpy as np

from scripts.native_language_cache import pair_key


def query_schedule(pairs, mapping, train_keys):
    if not pairs or set(pairs) != set(mapping) or not train_keys <= set(pairs):
        raise ValueError("complete matched query pairs/map and TRAIN keys required")
    groups = {"language/queries.npy": [], "new/queries.npy": []}
    for key, value in pairs.items():
        if not isinstance(value, list) or len(value) != 2 or key != pair_key(*value):
            raise ValueError("exact raw question/candidate pair hash required")
        row = mapping[key]
        if row["bank"] not in groups or type(row["offset"]) is not int or row["offset"] < 0:
            raise ValueError("known query bank and nonnegative offset required")
        groups[row["bank"]].append((row["offset"], key))
    for values in groups.values():
        if len({i for i, _ in values}) != len(values):
            raise ValueError("duplicate query offset")
    new = sorted(groups["new/queries.npy"])
    if [i for i, _ in new] != list(range(len(new))):
        raise ValueError("contiguous complete new query offsets required")
    smoke = [key for _, key in new if key in train_keys][:32]
    reference = [key for _, key in sorted(groups["language/queries.npy"]) if key in train_keys]
    if not smoke or not reference:
        raise ValueError("TRAIN-only missing query probe and existing reference required")
    return dict(all=[key for _, key in new], smoke=smoke, reference=reference[0])


def validate_vectors(values, count):
    if (
        not isinstance(values, np.ndarray)
        or values.shape != (count, 384)
        or values.dtype != np.float32
        or not np.isfinite(values).all()
    ):
        raise ValueError("complete finite FP32 candidate vectors required")
