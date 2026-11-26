"""Fixed temporal sampling for frozen slot features, without repeating frames."""

import numpy as np


def sample_indices(frame_count):
    if not isinstance(frame_count, int) or frame_count < 32:
        raise ValueError("at least 32 frames required")
    return np.linspace(0, frame_count - 1, 32).round().astype(int).tolist()


def pack_slots(slots, frame_ids, frame_count):
    slots = np.asarray(slots, dtype=np.float32)
    if slots.shape != (32, 7, 128) or not np.isfinite(slots).all():
        raise ValueError("finite [32,7,128] slots required")
    if frame_ids != sample_indices(frame_count):
        raise ValueError("sampling order changed")
    time = np.asarray(frame_ids, dtype=np.float32) / (frame_count - 1)
    time = np.broadcast_to(time[:, None, None], (32, 7, 1))
    return np.concatenate([slots, time], axis=-1).reshape(224, 129)
