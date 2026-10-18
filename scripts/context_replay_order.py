"""Fixed equal-source replay allocation, preserving the context order prefix."""

import random


def replay_order(context, original, steps=384):
    if steps <= 0 or len(context) < steps * 4 or len(original) < steps * 4:
        raise ValueError("insufficient unique source rows")
    for rows in (context, original):
        if len({r['id'] for r in rows}) != len(rows):
            raise ValueError("duplicate source IDs")
    if {r['id'] for r in context} & {r['id'] for r in original}:
        raise ValueError("source ID overlap")
    indices = list(range(len(original)))
    random.Random(17).shuffle(indices)
    result = []
    for step in range(steps):
        start = step * 4
        result.extend(dict(r, replay_source='context') for r in context[start:start + 4])
        result.extend(dict(original[i], replay_source='original')
                      for i in indices[start:start + 4])
    return result
