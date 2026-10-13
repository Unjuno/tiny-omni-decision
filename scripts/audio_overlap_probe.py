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
    numerator = convolution[n - 1 : len(b)]
    sums = np.r_[0.0, np.cumsum(b)]
    squares = np.r_[0.0, np.cumsum(b * b)]
    variance = squares[n:] - squares[:-n] - (sums[n:] - sums[:-n]) ** 2 / n
    valid = variance > 1e-12
    if not valid.any():
        return {"absolute_correlation": None, "offset_samples": None}
    scores = np.full(len(variance), -np.inf)
    scores[valid] = np.abs(numerator[valid]) / np.sqrt(energy * variance[valid])
    index = int(scores.argmax())
    return {"absolute_correlation": float(np.clip(scores[index], 0, 1)), "offset_samples": index}


class PreparedWaveformMatcher:
    """Reuse Fourier transforms for exhaustive whole-template comparisons.

    Uses the same normalization and valid offsets as peak_normalized_match.
    Preparation trades bounded host memory for fewer repeated transforms.
    """

    def __init__(self, waveforms, memory_limit_bytes=2 * 1024**3):
        if not waveforms:
            raise ValueError("nonempty waveform collection required")
        arrays = {k: np.asarray(v, dtype=np.float64) for k, v in waveforms.items()}
        if any(a.ndim != 1 or len(a) < 2 or not np.isfinite(a).all() for a in arrays.values()):
            raise ValueError("finite mono waveforms of at least two samples required")
        self.size = 1 << (2 * max(map(len, arrays.values())) - 2).bit_length()
        # Two complex spectra and two float64 prefix arrays per waveform.
        self.cache_bytes = sum(
            32 * (self.size // 2 + 1) + 16 * (len(a) + 1) for a in arrays.values()
        )
        if self.cache_bytes > memory_limit_bytes:
            raise ValueError("prepared waveform cache exceeds memory budget")
        self.data = {}
        for identity, a in arrays.items():
            centered = a - a.mean()
            self.data[identity] = (
                len(a),
                float(np.dot(centered, centered)),
                np.fft.rfft(centered[::-1], self.size),
                np.fft.rfft(a, self.size),
                np.r_[0.0, np.cumsum(a)],
                np.r_[0.0, np.cumsum(a * a)],
            )

    def match(self, template_id, recording_id):
        n, energy, template_fft, _, _, _ = self.data[template_id]
        length, _, _, recording_fft, sums, squares = self.data[recording_id]
        if length < n:
            raise ValueError("template must fit recording")
        if energy <= 1e-12:
            return {"absolute_correlation": None, "offset_samples": None}
        numerator = np.fft.irfft(recording_fft * template_fft, self.size)[n - 1 : length]
        variance = squares[n:] - squares[:-n] - (sums[n:] - sums[:-n]) ** 2 / n
        valid = variance > 1e-12
        if not valid.any():
            return {"absolute_correlation": None, "offset_samples": None}
        scores = np.full(len(variance), -np.inf)
        scores[valid] = np.abs(numerator[valid]) / np.sqrt(energy * variance[valid])
        index = int(scores.argmax())
        return {
            "absolute_correlation": float(np.clip(scores[index], 0, 1)),
            "offset_samples": index,
        }
