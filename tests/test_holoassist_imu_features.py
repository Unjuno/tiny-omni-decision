from __future__ import annotations

import numpy as np
import pytest

from scripts.build_holoassist_imu_features import (
    SENSORS,
    build_session_features,
    pool_temporal_features,
)


def write_sensor(path, *, values):
    path.write_text(
        "".join(
            f"{time}\t{1_000_000 + int(time * 1000)}\t{value}\t{value + 1}\t{value + 2}\n"
            for time, value in values
        ),
        encoding="utf-8",
    )


def test_sensor_features_are_binned_and_window_padding_is_fixed(tmp_path):
    files = {}
    for index, sensor in enumerate(SENSORS):
        path = tmp_path / f"{sensor}.txt"
        write_sensor(
            path,
            values=[(0.0, float(index)), (0.5, float(index + 2)), (1.0, 9.0), (1.1, 12.0)],
        )
        files[sensor] = path

    features = build_session_features(files, duration=2.0)
    assert features.shape == (2, 21)
    expected_means = [11 / 3 + axis for axis in range(3)]
    np.testing.assert_allclose(features[0, :3], expected_means)
    assert np.all(features[0, 18:21] == pytest.approx(3 / 30))
    np.testing.assert_allclose(features[1, :3], [12, 13, 14])

    window = pool_temporal_features(features, 0, 1)
    assert window.shape == (30, 21)
    np.testing.assert_array_equal(window[0], features[0])
    assert np.count_nonzero(window[1:]) == 0


def test_sensor_feature_parser_rejects_nonmonotone_timestamps(tmp_path):
    files = {}
    for sensor in SENSORS:
        path = tmp_path / f"{sensor}.txt"
        write_sensor(path, values=[(0.2, 1.0), (0.1, 2.0)])
        files[sensor] = path

    with pytest.raises(ValueError, match="non-finite or non-monotone"):
        build_session_features(files, duration=1.0)


def test_temporal_pool_rejects_windows_longer_than_fixed_horizon():
    features = np.zeros((60, 21), dtype=np.float32)
    with pytest.raises(ValueError, match="invalid temporal feature window"):
        pool_temporal_features(features, 0, 31)
