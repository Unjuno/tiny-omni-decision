"""Numeric source diagnostics without audio-channel, clock or label inference."""

import hashlib

import numpy as np


def summarize_stream(times, values):
    if (
        times.ndim != 2
        or times.shape[1] != 1
        or values.ndim != 2
        or times.shape[0] != values.shape[0]
        or len(times) < 2
        or not np.issubdtype(times.dtype, np.floating)
        or not np.issubdtype(values.dtype, np.floating)
        or not np.isfinite(times).all()
    ):
        raise ValueError("aligned numeric matrix rows and finite timestamp column required")
    delta = np.diff(times[:, 0])
    nonfinite = (~np.isfinite(values)).sum(axis=0).tolist()
    return dict(
        rows=len(times),
        channels=values.shape[1],
        time_dtype=str(times.dtype),
        value_dtype=str(values.dtype),
        first_time=float(times[0, 0]),
        last_time=float(times[-1, 0]),
        span_seconds=float(times[-1, 0] - times[0, 0]),
        timestamp_duplicates=int((delta == 0).sum()),
        timestamp_backwards=int((delta < 0).sum()),
        timestamp_delta_quantiles=[float(v) for v in np.quantile(delta, [0, 0.5, 1])],
        value_nonfinite=nonfinite,
        channel_minimum=[None if n else float(values[:, i].min()) for i, n in enumerate(nonfinite)],
        channel_maximum=[None if n else float(values[:, i].max()) for i, n in enumerate(nonfinite)],
        time_bytes_sha256=hashlib.sha256(np.ascontiguousarray(times).tobytes()).hexdigest(),
        values_bytes_sha256=hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest(),
        retimed=False,
        interpolated=False,
        calibrated=False,
        channel_semantics="UNVERIFIED",
        fine_clock_accuracy="UNVERIFIED",
    )
