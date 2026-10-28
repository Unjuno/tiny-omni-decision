from __future__ import annotations

import numpy as np
import pytest

from scripts.smoke_adl_audio_vibration_features import (
    participant_split,
    psutil_version,
    resample_vibration_axes,
)


def test_participant_split_is_stable_and_keeps_special_id_unassigned() -> None:
    assert participant_split(1) == participant_split(1)
    assert participant_split(999) is None
    assert participant_split(0) in {"train", "validation"}


def test_memory_probe_dependency_is_optional() -> None:
    version = psutil_version()

    assert version is None or isinstance(version, str)


def test_resample_vibration_axes_returns_fixed_finite_three_channel_window() -> None:
    rate = 1080.0
    time = np.arange(round(rate * 2.0), dtype=np.float64) / rate
    axes = [time, 2 * time, np.sin(time)]

    result = resample_vibration_axes(axes, rate)

    assert result.shape == (3, 512)
    assert result.dtype == np.float32
    assert np.isfinite(result).all()
    assert result[0, 0] == pytest.approx(0.0)
    assert result[1, -1] == pytest.approx(2 * (511 * 2 / 512), abs=1e-3)


@pytest.mark.parametrize(
    ("axes", "rate"),
    [
        ([np.zeros(2200)] * 2, 1080.0),
        ([np.zeros(100)] * 3, 1080.0),
        ([np.zeros(2200)] * 3, 0.0),
        ([np.zeros(2200), np.zeros(2200), np.full(2200, np.nan)], 1080.0),
    ],
)
def test_resample_vibration_axes_fails_closed(axes: list[np.ndarray], rate: float) -> None:
    with pytest.raises(ValueError):
        resample_vibration_axes(axes, rate)
