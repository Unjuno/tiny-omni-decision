"""Fixed eight-frame selection confined to declared past-window bounds."""

import math
from bisect import bisect_left, bisect_right


def past_window_indices(timestamps_ms, *, start_ms, end_ms):
    """Sample eight distinct indices in inclusive [start,end], never later.

    This selects already-decoded presentation timestamps, not an online capture
    API. Window input completion and encoder completion remain separate events.
    """
    stamps = tuple(timestamps_ms)
    if not math.isfinite(start_ms) or not math.isfinite(end_ms) or start_ms >= end_ms:
        raise ValueError("finite increasing window bounds required")
    if (not stamps or any(not math.isfinite(t) for t in stamps)
            or any(b <= a for a, b in zip(stamps, stamps[1:], strict=False))):
        raise ValueError("finite strictly increasing timestamps required")
    first, stop = bisect_left(stamps, start_ms), bisect_right(stamps, end_ms)
    if stop - first < 8:
        raise ValueError("at least eight distinct frames needed within window")
    return [first + round(i * (stop - first - 1) / 7) for i in range(8)]
