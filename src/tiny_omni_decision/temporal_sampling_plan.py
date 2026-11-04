"""Prospective dense sampling plan; does not replace existing input policies."""

import math
from bisect import bisect_right


def plan_windows(timestamps, *, start, end, fps=4, width=8):
    stamps = tuple(timestamps)
    if (
        not stamps
        or any(not math.isfinite(t) for t in stamps)
        or any(b <= a for a, b in zip(stamps, stamps[1:], strict=False))
    ):
        raise ValueError("strictly increasing finite timestamps required")
    if not (
        math.isfinite(start)
        and math.isfinite(end)
        and start < end
        and math.isfinite(fps)
        and fps > 0
        and width >= 2
    ):
        raise ValueError("invalid sampling bounds")
    indices = []
    for step in range(math.floor((end - start) * fps) + 1):
        target = start + step / fps
        index = bisect_right(stamps, target) - 1
        if index >= 0 and start <= stamps[index] < end:
            if not indices or indices[-1] != index:
                indices.append(index)
    last = bisect_right(stamps, math.nextafter(end, -math.inf)) - 1
    if last >= 0 and start <= stamps[last] < end and (not indices or indices[-1] != last):
        indices.append(last)
    if len(indices) < width:
        raise ValueError("insufficient distinct frames")
    windows = [indices[i : i + width] for i in range(0, len(indices) - width + 1, width)]
    if windows[-1][-1] != indices[-1]:
        windows.append(indices[-width:])
    selected = [stamps[i] for i in indices]
    return dict(
        indices=indices,
        windows=windows,
        max_gap_seconds=max(b - a for a, b in zip(selected, selected[1:], strict=False)),
        boundary_gaps_seconds=[selected[0] - start, end - selected[-1]],
        repeated_window_frames=sum(map(len, windows)) - len(indices),
    )
