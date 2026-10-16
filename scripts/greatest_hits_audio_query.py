"""Label-independent query inventory and exact-once native audio bank writes."""

import numpy as np

from scripts.greatest_hits_cache import frozen_events
from scripts.native_language_cache import pair_key


def query_inventory(splits):
    frozen_events(splits)
    pairs = {}
    for rows in splits.values():
        for row in rows:
            for candidate in row['options']:
                key = pair_key(row['question'], candidate)
                item = dict(key=key, question=row['question'], candidate=candidate)
                if key in pairs and pairs[key] != item:
                    raise ValueError('query key collision')
                pairs[key] = item
    return [pairs[k] for k in sorted(pairs)]


def audio_positions():
    speech = np.zeros((25, 3), np.float32)
    speech[:, 0] = np.arange(25) * .02 / .5
    return dict(speech=speech, event_audio=np.array([[.5, 0, 0]], np.float32))


def store_audio_event(banks, index, values):
    shapes = dict(speech=(25, 384), event_audio=(1, 384))
    if set(banks) != set(shapes) or set(values) != set(shapes):
        raise ValueError('both native audio streams required')
    counts = set()
    for kind, shape in shapes.items():
        bank, value = banks[kind], values[kind]
        if (not isinstance(bank, np.ndarray) or bank.ndim != 3
                or bank.shape[1:] != shape or bank.dtype != np.float32
                or not bank.flags.writeable or type(index) is not int
                or not 0 <= index < len(bank) or not isinstance(value, np.ndarray)
                or value.shape != shape or value.dtype != np.float32
                or not np.isfinite(value).all()):
            raise ValueError('finite FP32 native shapes and valid bank index required')
        counts.add(len(bank))
        if not np.isnan(bank[index]).all():
            raise ValueError('bank event already written')
    if len(counts) != 1:
        raise ValueError('audio banks must have the same event count')
    for kind, value in values.items():
        banks[kind][index] = value
