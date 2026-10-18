"""Source-free frozen video-memory probe routing; no semantic target selection."""

import numpy as np


def seed_masks(packed, indices):
    if (
        not isinstance(packed, np.ndarray)
        or packed.ndim != 2
        or packed.shape[1] != 8192
        or packed.dtype != np.uint8
        or not isinstance(indices, np.ndarray)
        or indices.dtype != np.int64
        or indices.ndim != 1
        or not len(indices)
        or (indices < 0).any()
        or (indices >= len(packed)).any()
        or len(np.unique(indices)) != len(indices)
    ):
        raise ValueError("complete native256 packed masks and unique NMS indices required")
    result = (
        np.unpackbits(packed[indices], axis=1, bitorder="big").reshape(-1, 256, 256).astype(bool)
    )
    if not result.reshape(len(result), -1).any(1).all():
        raise ValueError("an existing NMS seed is empty; never silently filter it")
    return result


def video_control(rgb, arm):
    if (
        not isinstance(rgb, np.ndarray)
        or rgb.ndim != 4
        or min(rgb.shape[:3]) < 1
        or rgb.shape[-1] != 3
        or rgb.dtype != np.uint8
        or arm not in {"actual", "initial_repeat"}
    ):
        raise ValueError("complete native RGB sequence and explicit control required")
    return rgb.copy() if arm == "actual" else np.repeat(rgb[:1], len(rgb), axis=0)


def pack_frame(logits, score):
    if (
        not isinstance(logits, np.ndarray)
        or logits.shape != (1, 1, 256, 256)
        or logits.dtype != np.float32
        or not np.isfinite(logits).all()
        or not isinstance(score, np.ndarray)
        or score.shape != (1, 1)
        or score.dtype != np.float32
        or not np.isfinite(score).all()
    ):
        raise ValueError("finite native single-object FP32 mask/score required")
    mask = logits[0, 0] > 0
    packed = np.packbits(mask.reshape(-1), bitorder="big")
    y, x = np.nonzero(mask)
    box = (
        np.array(
            [x.min() / 256, y.min() / 256, (x.max() + 1) / 256, (y.max() + 1) / 256], np.float64
        )
        if len(x)
        else np.zeros(4, np.float64)
    )
    return packed, box


def verify_times(frames, original_frames, cut):
    if (
        type(original_frames) is not int
        or original_frames < 32
        or type(cut) is not int
        or cut not in {-1, original_frames}
        or not isinstance(frames, np.ndarray)
        or frames.dtype != np.int64
        or frames.shape != (32,)
        or not np.array_equal(
            frames, np.rint(np.linspace(0, original_frames - 1, 32)).astype(np.int64)
        )
    ):
        raise ValueError("unchanged32 original source times/cut required")
