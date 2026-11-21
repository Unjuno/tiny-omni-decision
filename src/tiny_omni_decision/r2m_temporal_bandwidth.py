"""Audit temporal bandwidth lost when variable-length vibration is tokenized."""

from __future__ import annotations

import math

import numpy as np


def effective_uniform_sample_rate(sample_count: int, duration_seconds: float) -> float:
    """Return the rate of a grid including both endpoints of a duration."""
    if type(sample_count) is not int or sample_count < 2:
        raise ValueError("sample_count must be an integer >= 2")
    if not math.isfinite(duration_seconds) or duration_seconds <= 0:
        raise ValueError("duration_seconds must be finite and positive")
    return (sample_count - 1) / duration_seconds


def out_of_band_energy_fraction(
    signal: np.ndarray,
    source_rate_hz: float,
    target_rate_hz: float,
) -> float:
    """Measure demeaned signal energy above the target grid's Nyquist rate.

    This is a raw-spectrum diagnostic, not an estimate of downstream model
    accuracy or retained information after linear interpolation.
    """
    values = np.asarray(signal, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise ValueError("signal must be a finite one-dimensional array of length >= 2")
    if (
        not math.isfinite(source_rate_hz)
        or source_rate_hz <= 0
        or not math.isfinite(target_rate_hz)
        or target_rate_hz <= 0
    ):
        raise ValueError("sample rates must be finite and positive")
    centered = values - values.mean()
    spectrum = np.fft.rfft(centered)
    energy = np.square(np.abs(spectrum))
    total = float(energy.sum())
    if total <= 0:
        return 0.0
    frequencies = np.fft.rfftfreq(values.size, d=1.0 / source_rate_hz)
    return float(energy[frequencies > target_rate_hz / 2].sum() / total)
