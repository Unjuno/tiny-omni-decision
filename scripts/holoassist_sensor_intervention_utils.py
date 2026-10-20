"""Numpy-only helpers for controlled HoloAssist sensor interventions."""

from __future__ import annotations

from typing import Any

import numpy as np


def derange_sensor_sessions(data: dict[str, Any]) -> tuple[dict[str, Any], dict[str, int]]:
    """Rotate whole IMU sequences within exact task and session-length groups."""
    result = dict(data)
    result["sensor"] = np.asarray(data["sensor"]).copy()
    groups: dict[tuple[str, int], list[int]] = {}
    for index, indices in enumerate(data["session_indices"]):
        first = int(indices[0])
        key = (str(data["task_type"][first]), len(indices))
        groups.setdefault(key, []).append(index)
    changed = 0
    eligible = 0
    for members in groups.values():
        members.sort(key=lambda item: str(data["session_names"][item]))
        if len(members) < 2:
            continue
        eligible += len(members)
        donor_map = {
            member: members[(position + 1) % len(members)]
            for position, member in enumerate(members)
        }
        for receiver in members:
            receiver_rows = data["session_indices"][receiver]
            donor_rows = data["session_indices"][donor_map[receiver]]
            if np.array_equal(data["sensor"][receiver_rows], data["sensor"][donor_rows]):
                continue
            result["sensor"][receiver_rows] = data["sensor"][donor_rows]
            changed += 1
    return result, {
        "sessions_total": len(data["session_indices"]),
        "sessions_eligible": eligible,
        "sessions_changed": changed,
        "singleton_task_length_groups": sum(
            len(members) for members in groups.values() if len(members) < 2
        ),
    }
