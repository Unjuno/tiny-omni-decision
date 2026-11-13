"""Load a hash-bound sensor NPZ with explicit window identity and no target fields."""

import hashlib
import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tiny_omni_decision.observations import Observation


@dataclass(frozen=True)
class SensorPayload:
    values: np.ndarray
    timestamps_ms: np.ndarray
    valid_mask: np.ndarray
    units: tuple[str, ...]


def load_sensor_payload(observation: Observation, *, max_bytes=256 * 1024**2) -> SensorPayload:
    # Revalidate even model_copy(update=...) objects at this trust boundary.
    observation = Observation.model_validate(observation.model_dump())
    if observation.modality != "sensor" or max_bytes < 1:
        raise ValueError("sensor observation and positive size budget required")
    with Path(observation.payload_ref).open("rb") as stream:
        raw = stream.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise ValueError("payload exceeds byte budget")
    if hashlib.sha256(raw).hexdigest() != observation.payload_sha256:
        raise ValueError("payload hash mismatch")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        names = archive.namelist()
        if sorted(names) != ["timestamps_ms.npy", "values.npy"]:
            raise ValueError("sensor payload must contain only values and timestamps")
        if sum(i.file_size for i in archive.infolist()) > max_bytes:
            raise ValueError("expanded payload exceeds byte budget")
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        values = archive["values"]
        times = archive["timestamps_ms"]
    if values.dtype.kind not in "fiu" or times.dtype.kind not in "fiu":
        raise ValueError("sensor arrays must be numeric")
    index = observation.payload_index
    if values.ndim == 3:
        if times.shape != values.shape[:2] or index is None or index >= len(values):
            raise ValueError("batched sensor payload needs aligned times and explicit valid index")
        values, times = values[index], times[index]
    elif values.ndim != 2 or index is not None:
        raise ValueError("unexpected sensor rank or index on unbatched payload")
    if values.shape != (len(observation.timestamps_ms), len(observation.channels)):
        raise ValueError("sensor payload shape mismatch")
    if not np.array_equal(times, np.asarray(observation.timestamps_ms, dtype=np.float64)):
        raise ValueError("sensor payload timestamps mismatch")
    valid = np.asarray(observation.valid_mask, dtype=bool)
    if not np.isfinite(values[valid]).all():
        raise ValueError("nonfinite valid sensor values")
    calibrated = np.zeros(values.shape, dtype=np.float64)
    scales = np.array([c.scale for c in observation.channels])
    offsets = np.array([c.offset for c in observation.channels])
    with np.errstate(over="ignore", invalid="ignore"):
        calibrated[valid] = values[valid].astype(np.float64) * scales + offsets
    if not np.isfinite(calibrated).all():
        raise ValueError("nonfinite calibrated values")
    for i, channel in enumerate(observation.channels):
        if channel.valid_range is not None:
            low, high = channel.valid_range
            if ((calibrated[valid, i] < low) | (calibrated[valid, i] > high)).any():
                raise ValueError("calibrated sensor value outside declared range")
    times = times.astype(np.float64, copy=True)
    for array in (calibrated, times, valid):
        array.setflags(write=False)
    return SensorPayload(calibrated, times, valid, tuple(c.unit for c in observation.channels))
