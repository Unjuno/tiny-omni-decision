"""Bounded-memory endpoint sampling with actual presentation timestamps."""

import math


def decode_endpoint_frames(path):
    """Decode twice, counting actual frames instead of trusting container hints.

    Require eight distinct frames and increasing presentation times. These times
    describe the video stream only; no sensor clock alignment is inferred.
    """
    import av
    import numpy as np

    timestamps = []
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError("no video stream")
        for frame in container.decode(container.streams.video[0]):
            if frame.time is None or not math.isfinite(frame.time):
                raise ValueError("missing/nonfinite video presentation timestamp")
            timestamps.append(float(frame.time) * 1000)
    if len(timestamps) < 8 or any(
        b <= a for a, b in zip(timestamps, timestamps[1:], strict=False)
    ):
        raise ValueError("eight frames and strictly increasing video times required")
    indices = [round(i * (len(timestamps) - 1) / 7) for i in range(8)]
    selected = []
    with av.open(str(path)) as container:
        for index, frame in enumerate(container.decode(container.streams.video[0])):
            if index in indices:
                if frame.time is None or float(frame.time) * 1000 != timestamps[index]:
                    raise ValueError("video changed between decoding passes")
                selected.append(frame.to_ndarray(format="rgb24"))
    if len(selected) != 8:
        raise ValueError("inconsistent video decode")
    return np.stack(selected), {"decoded_frame_count": len(timestamps), "indices": indices,
                                "timestamps_ms": [timestamps[i] for i in indices]}
