"""Retain complete time-indexed mask statistics without labels or models."""

import numpy as np


def require_aligned_feature_times(native_plan, feature_plan, times):
    """Reject an asset/cut/time mismatch before area pooling frozen grids."""
    keys = ("video_id", "split", "frames", "cut_frame", "media_path", "media_sha256")
    if (
        any(k not in native_plan or k not in feature_plan for k in keys)
        or any(native_plan[k] != feature_plan[k] for k in keys)
        or native_plan["split"] != "train"
        or type(native_plan["frames"]) is not int
        or native_plan["frames"] < 1
        or type(native_plan["cut_frame"]) is not int
        or native_plan["cut_frame"] not in (-1, native_plan["frames"])
    ):
        raise ValueError("identical approved training asset and source cut required")
    indices = native_plan.get("frame_indices")
    if (
        not isinstance(indices, list)
        or not indices
        or any(type(i) is not int for i in indices)
        or not _valid_times(times)
        or times.tolist() != indices
        or times[-1] >= native_plan["frames"]
        or indices
        != np.rint(np.linspace(0, native_plan["frames"] - 1, len(indices)))
        .astype(np.int64)
        .tolist()
    ):
        raise ValueError("complete identical uniform native observation times required")


def _valid_times(times):
    return (
        isinstance(times, np.ndarray)
        and times.dtype == np.int64
        and times.ndim == 1
        and len(times) > 0
        and times[0] >= 0
        and np.all(np.diff(times) > 0)
    )


def assemble_mask_statistics(observation_times, frames):
    """Keep all native survivor order, empty masks and even empty frame offsets."""
    keys = {"mean", "std", "foreground_counts", "proposal_ids", "boxes", "scores"}
    if (
        not _valid_times(observation_times)
        or not isinstance(frames, (list, tuple))
        or len(frames) != len(observation_times)
    ):
        raise ValueError("one ordered frame inventory per observation time required")
    channels, offsets, collections, per_frame_times = None, [0], {k: [] for k in keys}, []
    for time, frame in zip(observation_times, frames, strict=True):
        if not isinstance(frame, dict) or set(frame) != keys:
            raise ValueError("complete statistics-only frame inventory required")
        mean, std = frame["mean"], frame["std"]
        if (
            not isinstance(mean, np.ndarray)
            or mean.ndim != 2
            or mean.shape[1] < 1
            or mean.dtype != np.float32
            or not isinstance(std, np.ndarray)
            or std.shape != mean.shape
            or std.dtype != np.float32
        ):
            raise ValueError("matching FP32 mean/std channels required")
        n, c = mean.shape
        if channels is not None and c != channels:
            raise ValueError("feature channels changed across frames")
        channels = c
        for key, shape, dtype in [
            ("foreground_counts", (n,), np.int64),
            ("proposal_ids", (n,), np.int64),
            ("boxes", (n, 4), np.float64),
            ("scores", (n,), np.float32),
        ]:
            a = frame[key]
            if not isinstance(a, np.ndarray) or a.shape != shape or a.dtype != dtype:
                raise ValueError("frame array shape/dtype changed: " + key)
        if (
            any(not np.isfinite(a).all() for a in frame.values())
            or np.any(std < 0)
            or np.any(frame["foreground_counts"] < 0)
            or np.any(frame["proposal_ids"] < 0)
            or len(np.unique(frame["proposal_ids"])) != n
        ):
            raise ValueError("finite nonnegative statistics and unique native frame IDs required")
        empty = frame["foreground_counts"] == 0
        if np.any(mean[empty]) or np.any(std[empty]):
            raise ValueError("empty masks must retain zero descriptors and zero mass")
        for key, a in frame.items():
            collections[key].append(a)
        offsets.append(offsets[-1] + n)
        per_frame_times.append(np.full(n, time, np.int64))
    result = {k: np.concatenate(v) for k, v in collections.items()}
    result.update(
        frame_indices=np.concatenate(per_frame_times),
        frame_offsets=np.array(offsets, np.int64),
        observation_times=observation_times.copy(),
    )
    return result
