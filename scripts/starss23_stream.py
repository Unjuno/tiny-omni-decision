"""Bounded, single-open STARSS23 recording readers for frozen-window features."""

from __future__ import annotations

import math
import wave


def validate_intervals(windows):
    previous_end = -1
    for start, end in windows:
        if not all(math.isfinite(t) for t in (start, end)) or start < 0 or end - start != 5:
            raise ValueError("expected finite nonnegative five-second windows")
        if start < previous_end:
            raise ValueError("windows overlap or are out of order")
        previous_end = end
    if not windows:
        raise ValueError("no windows supplied")


def audio_windows(path, windows):
    import numpy as np

    validate_intervals(windows)
    with wave.open(str(path)) as source:
        if (source.getnchannels(), source.getframerate(), source.getsampwidth()) != (4, 24000, 2):
            raise ValueError("unexpected audio format")
        for start, end in windows:
            first, last = round(start * 24000), round(end * 24000)
            if last > source.getnframes():
                raise ValueError("audio window extends beyond recording")
            source.setpos(first)
            samples = np.frombuffer(source.readframes(last - first), dtype="<i2").reshape(-1, 4)
            if samples.shape != (120000, 4):
                raise ValueError("incomplete audio window")
            yield samples.astype(np.float32) / 32768


def video_windows(path, windows):
    import av
    import numpy as np

    validate_intervals(windows)
    index = 0
    selected, timestamps = [], []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if stream.start_time not in (None, 0):
            raise ValueError("unsupported video timeline origin")
        previous_timestamp = -float("inf")
        for frame in container.decode(stream):
            if frame.pts is None:
                raise ValueError("missing video timestamp")
            timestamp = float(frame.pts * stream.time_base)
            if timestamp < previous_timestamp:
                raise ValueError("nonmonotonic video presentation timestamps")
            previous_timestamp = timestamp
            start, end = windows[index]
            target = start + (len(selected) + 0.5) * 5 / 8
            if timestamp < target:
                continue
            if timestamp - target > 0.05 or not start <= timestamp < end:
                raise ValueError("video timestamp outside fixed sampling tolerance")
            selected.append(frame.to_ndarray(format="rgb24"))
            timestamps.append(timestamp)
            if len(selected) == 8:
                result = np.stack(selected), np.array(timestamps)
                selected, timestamps = [], []
                yield result
                del result
                index += 1
                if index == len(windows):
                    return
    raise ValueError("incomplete video windows")
