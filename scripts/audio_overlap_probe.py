"""Gain/offset-invariant waveform matching; not a semantic duplicate detector."""

import numpy as np


def peak_normalized_match(template, recording):
    a, b = np.asarray(template, dtype=np.float64), np.asarray(recording, dtype=np.float64)
    if a.ndim != 1 or b.ndim != 1 or len(a) < 2 or len(b) < len(a):
        raise ValueError("nonempty mono template must fit recording")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("finite waveforms required")
    centered = a - a.mean()
    energy = np.dot(centered, centered)
    if energy <= 1e-12:
        return {"absolute_correlation": None, "offset_samples": None}
    n = len(a)
    size = 1 << (len(a) + len(b) - 2).bit_length()
    convolution = np.fft.irfft(np.fft.rfft(b, size) * np.fft.rfft(centered[::-1], size), size)
    numerator = convolution[n - 1:len(b)]
    sums = np.r_[0.0, np.cumsum(b)]
    squares = np.r_[0.0, np.cumsum(b * b)]
    variance = squares[n:] - squares[:-n] - (sums[n:] - sums[:-n]) ** 2 / n
    valid = variance > 1e-12
    if not valid.any():
        return {"absolute_correlation": None, "offset_samples": None}
    scores = np.full(len(variance), -np.inf)
    scores[valid] = np.abs(numerator[valid]) / np.sqrt(energy * variance[valid])
    index = int(scores.argmax())
    return {"absolute_correlation": float(np.clip(scores[index], 0, 1)),
            "offset_samples": index}
