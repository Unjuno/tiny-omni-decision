"""Strict processed-quaternion reader; no inferred clocks or resampling."""

import numpy as np


def parse_quaternion_sto(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if "endheader" not in lines:
        raise ValueError("missing STO header")
    end = lines.index("endheader")
    if "DataType=Quaternion" not in lines[:end] or len(lines) < end + 4:
        raise ValueError("quaternion header and at least two samples required")
    columns = lines[end + 1].split()
    sensors = tuple(columns[1:])
    if columns[0] != "time" or not sensors or len(set(sensors)) != len(sensors):
        raise ValueError("invalid sensor columns")
    times, values = [], []
    for line in lines[end + 2:]:
        cells = line.split()
        if len(cells) != len(columns):
            raise ValueError("row width mismatch")
        times.append(float(cells[0]))
        row = [list(map(float, cell.split(","))) for cell in cells[1:]]
        if any(len(q) != 4 for q in row):
            raise ValueError("four quaternion components required")
        values.append(row)
    times = np.asarray(times, dtype=np.float64) * 1000
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(times).all() or not np.isfinite(values).all():
        raise ValueError("nonfinite observation")
    if np.any(np.diff(times) <= 0):
        raise ValueError("timestamps must increase")
    if not np.allclose(np.linalg.norm(values, axis=-1), 1, atol=1e-3, rtol=0):
        raise ValueError("quaternion norm outside tolerance")
    times.setflags(write=False)
    values.setflags(write=False)
    return {"timestamps_ms": times, "values": values, "sensors": sensors,
            "component_order": ("w", "x", "y", "z"), "unit": "dimensionless"}
