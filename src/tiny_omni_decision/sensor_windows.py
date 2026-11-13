"""Explicit gap/mask-aware sensor windows; never interpolate absent observations."""

import math


def contiguous_windows(timestamps_ms, valid_mask, *, length, stride, max_gap_ms):
    if (
        not isinstance(length, int)
        or isinstance(length, bool)
        or length < 1
        or not isinstance(stride, int)
        or isinstance(stride, bool)
        or stride < 1
        or not math.isfinite(max_gap_ms)
        or max_gap_ms <= 0
    ):
        raise ValueError("window sizes and maximum gap must be positive")
    if len(timestamps_ms) != len(valid_mask):
        raise ValueError("time/mask length mismatch")
    if any(not isinstance(v, bool) for v in valid_mask):
        raise ValueError("validity must be boolean")
    if any(not math.isfinite(t) for t in timestamps_ms):
        raise ValueError("nonfinite timestamp")
    if any(b <= a for a, b in zip(timestamps_ms, timestamps_ms[1:], strict=False)):
        raise ValueError("timestamps must strictly increase")
    output = []
    start = None
    for i in range(len(timestamps_ms) + 1):
        valid = i < len(timestamps_ms) and valid_mask[i]
        gap = (
            i > 0
            and i < len(timestamps_ms)
            and timestamps_ms[i] - timestamps_ms[i - 1] > max_gap_ms
        )
        if start is not None and (not valid or gap):
            output.extend((j, j + length) for j in range(start, i - length + 1, stride))
            start = None
        if valid and start is None:
            start = i
    return output
