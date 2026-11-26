"""Prospectively fixed long curve; no outcome-dependent stop or budget extension."""

import hashlib
import json
import math

MAX_STEPS = 32768
EVALUATION_STEPS = (0, 512, 1024, 2048, 4096, 8192, 12288, 16384, 24576, 32768)


def learning_rate(step):
    if type(step) is not int or not 1 <= step <= MAX_STEPS:
        raise ValueError("fixed32768-step schedule required")
    if step <= 62:
        return 0.001 * step / 62
    return 0.0005 * (1 + math.cos(math.pi * (step - 62) / (MAX_STEPS - 62)))


def extend_plan(pool, batches, max_steps=MAX_STEPS):
    if (
        not pool
        or not batches
        or type(max_steps) is not int
        or max_steps < len(batches)
        or max_steps % len(batches)
        or any(type(i) is not int or i < 0 for i in pool)
        or any(len(b) != 8 for b in batches)
        or any(type(o) is not int for b in batches for o in b)
        or sorted(o for b in batches for o in b) != list(range(len(pool)))
    ):
        raise ValueError("complete eight-microbatch occurrence plan and whole cycles required")
    cycles = max_steps // len(batches)
    order = [[pool[o] for o in b] for _ in range(cycles) for b in batches]
    occurrences = [[o + cycle * len(pool) for o in b] for cycle in range(cycles) for b in batches]
    payload = dict(order=order, occurrences=occurrences)
    return dict(
        **payload,
        cycles=cycles,
        unique_row_indices=sorted(set(pool)),
        order_sha256=hashlib.sha256(
            json.dumps(payload, separators=(",", ":")).encode()
        ).hexdigest(),
    )
