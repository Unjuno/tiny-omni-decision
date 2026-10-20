from __future__ import annotations

import numpy as np

from scripts.evaluate_holoassist_av_sensor_interventions import derange_sensor_sessions


def _fixture() -> dict[str, object]:
    return {
        "session_names": np.asarray(["s1", "s2", "s3", "s4"]),
        "task_type": np.asarray(
            ["repair", "repair", "repair", "repair", "repair", "repair", "setup"]
        ),
        "session_indices": [
            np.asarray([0, 1]),
            np.asarray([2, 3]),
            np.asarray([4, 5]),
            np.asarray([6]),
        ],
        "sensor": np.arange(7 * 3, dtype=np.float32).reshape(7, 3),
    }


def test_derange_sensor_sessions_rotates_whole_matching_sequences() -> None:
    data = _fixture()

    altered, counts = derange_sensor_sessions(data)

    assert counts == {
        "sessions_total": 4,
        "sessions_eligible": 3,
        "sessions_changed": 3,
        "singleton_task_length_groups": 1,
    }
    assert np.array_equal(altered["sensor"][:2], data["sensor"][2:4])
    assert np.array_equal(altered["sensor"][2:4], data["sensor"][4:6])
    assert np.array_equal(altered["sensor"][4:6], data["sensor"][:2])
    assert np.array_equal(altered["sensor"][6:], data["sensor"][6:])
    assert np.array_equal(data["sensor"], np.arange(21, dtype=np.float32).reshape(7, 3))


def test_derangement_is_deterministic_and_never_crosses_task_or_length() -> None:
    data = _fixture()

    first, _ = derange_sensor_sessions(data)
    second, _ = derange_sensor_sessions(data)

    assert np.array_equal(first["sensor"], second["sensor"])
    assert np.array_equal(first["sensor"][6:], data["sensor"][6:])
