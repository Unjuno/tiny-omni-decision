"""Deterministic eight-frame windows for cached VideoPrism token extraction."""

from __future__ import annotations

from collections.abc import Sequence


def plan_frame_windows(
    frame_ids: Sequence[int], *, window_size: int = 8, stride: int = 4
) -> dict[str, object]:
    """Cover ordered annotated frame IDs with overlapping windows and a stable slot map."""
    values = list(frame_ids)
    if (
        not values
        or any(type(frame_id) is not int or frame_id < 0 for frame_id in values)
        or any(left >= right for left, right in zip(values, values[1:], strict=False))
    ):
        raise ValueError("frame IDs must be nonempty, unique increasing nonnegative integers")
    if type(window_size) is not int or window_size <= 0:
        raise ValueError("window size must be a positive integer")
    if type(stride) is not int or stride <= 0 or stride > window_size:
        raise ValueError("stride must be a positive integer no greater than window size")

    count = len(values)
    if count <= window_size:
        starts = [0]
    else:
        last_start = count - window_size
        starts = list(range(0, last_start + 1, stride))
        if starts[-1] != last_start:
            starts.append(last_start)

    windows = []
    for start in starts:
        selected = list(values[start : start + window_size])
        selected.extend([values[-1]] * (window_size - len(selected)))
        windows.append(selected)

    frame_to_window_slot = {}
    center_offset = (window_size - 1) / 2
    for position, frame_id in enumerate(values):
        eligible = [
            (abs(start + center_offset - position), window_index, position - start)
            for window_index, start in enumerate(starts)
            if start <= position < min(start + window_size, count)
        ]
        _, window_index, slot_index = min(eligible)
        frame_to_window_slot[frame_id] = (window_index, slot_index)

    return {"windows": windows, "frame_to_window_slot": frame_to_window_slot}
