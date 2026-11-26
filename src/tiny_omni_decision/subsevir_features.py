"""Small pure transforms for frozen sub-SEVIR frame features."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

STREAM_CHANNELS: dict[str, tuple[int, int, int]] = {
    "satellite": (0, 1, 3),
    "radar": (2, 2, 2),
}


def frame_to_rgb(
    frame: np.ndarray,
    percentile_ranges: Sequence[Sequence[float]],
    stream: str,
) -> np.ndarray:
    """Map one HWC sub-SEVIR frame into deterministic uint8 pseudo-RGB."""
    if stream not in STREAM_CHANNELS:
        raise ValueError(f"unsupported stream: {stream}")
    if frame.ndim != 3 or frame.shape[-1] != 4:
        raise ValueError("frame must be HWC with four sub-SEVIR channels")
    if len(percentile_ranges) != 4 or any(len(bounds) != 2 for bounds in percentile_ranges):
        raise ValueError("expected four train-only [p01, p99] ranges")
    rgb_channels = []
    for channel in STREAM_CHANNELS[stream]:
        low, high = map(float, percentile_ranges[channel])
        if not np.isfinite([low, high]).all() or high <= low:
            raise ValueError(f"invalid range for source channel {channel}")
        values = np.clip((frame[..., channel] - low) / (high - low), 0.0, 1.0)
        rgb_channels.append(np.rint(values * 255).astype(np.uint8))
    result = np.stack(rgb_channels, axis=-1)
    if not np.isfinite(result).all():
        raise ValueError("non-finite pseudo-RGB frame")
    return result


def required_frame_indices(windows: Sequence[Mapping[str, object]]) -> dict[str, list[int]]:
    """Return sorted, unique source rows referenced by each frozen split."""
    result: dict[str, set[int]] = {"train": set(), "validation": set()}
    event_splits: dict[str, str] = {}
    for window in windows:
        split = str(window["split"])
        if split not in result:
            raise ValueError(f"unexpected split: {split}")
        event_id = str(window["event_id"])
        previous = event_splits.setdefault(event_id, split)
        if previous != split:
            raise ValueError(f"event crosses splits: {event_id}")
        indices = window["input_indices"]
        if not isinstance(indices, list) or len(indices) != 8:
            raise ValueError("each frozen example must contain exactly eight input indices")
        result[split].update(int(index) for index in indices)
    return {split: sorted(indices) for split, indices in result.items()}


def cls_plus_patch_mean(last_hidden_state: np.ndarray) -> np.ndarray:
    """Build an orderable per-frame vector from the audited DINOv2 tokens."""
    if last_hidden_state.ndim != 3 or last_hidden_state.shape[1] < 2:
        raise ValueError("expected [batch, CLS+patch tokens, hidden size]")
    cls = last_hidden_state[:, 0, :]
    patch_mean = last_hidden_state[:, 1:, :].mean(axis=1)
    result = np.concatenate((cls, patch_mean), axis=-1).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("non-finite frame features")
    return result


def train_event_percentile_ranges(group: object, train_event_ids: set[str]) -> list[list[float]]:
    """Estimate per-channel p01/p99 using only frozen training-event rows.

    For bounded memory and reproducibility, sample spatial positions with a
    fixed stride of 97 within each frame. The Zarr group is read only.
    """
    arrays = group  # Keep the generic type annotation friendly to the optional Zarr runtime.
    event_ids = [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in arrays["id"][:]
    ]
    selected = np.fromiter((event_id in train_event_ids for event_id in event_ids), dtype=bool)
    if not selected.any():
        raise ValueError("no rows matched frozen training event IDs")
    features = arrays["features"]
    channel_samples: list[list[np.ndarray]] = [[], [], [], []]
    for start in range(0, len(event_ids), features.chunks[0]):
        stop = min(start + features.chunks[0], len(event_ids))
        mask = selected[start:stop]
        if not mask.any():
            continue
        block = np.asarray(features[start:stop], dtype=np.float32).reshape(stop - start, 2304, 4)
        sampled = block[mask, ::97, :]
        if not np.isfinite(sampled).all():
            raise ValueError("non-finite training feature values")
        for channel in range(4):
            channel_samples[channel].append(sampled[..., channel].reshape(-1))
    ranges = []
    for samples in channel_samples:
        joined = np.concatenate(samples)
        low, high = np.percentile(joined, [1.0, 99.0]).astype(float).tolist()
        if not np.isfinite([low, high]).all() or high <= low:
            raise ValueError("degenerate train-only normalization range")
        ranges.append([low, high])
    return ranges
