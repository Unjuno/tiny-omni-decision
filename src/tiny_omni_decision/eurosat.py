"""Integrity checks for pinned EuroSAT split manifests."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import PurePosixPath

import numpy as np

SPLIT_NAMES = ("train", "validation", "test")
HOTGPU_ENV_KEYS = (
    "TINY_OMNI_GPU_GATE_RESERVED_MIB",
    "TINY_OMNI_GPU_GATE_FREE_MIB",
    "TINY_OMNI_GPU_GATE_TOTAL_MIB",
)


def require_hotgpu_reservation(environment: Mapping[str, str]) -> dict[str, int]:
    """Require valid reservation telemetry before starting a GPU evaluation."""
    values = {}
    for key in HOTGPU_ENV_KEYS:
        try:
            values[key] = int(environment[key])
        except (KeyError, ValueError) as error:
            message = f"evaluation must be launched through HotUGPU; invalid {key}"
            raise RuntimeError(message) from error
    if (
        values[HOTGPU_ENV_KEYS[0]] <= 0
        or values[HOTGPU_ENV_KEYS[1]] <= 0
        or values[HOTGPU_ENV_KEYS[2]] < values[HOTGPU_ENV_KEYS[1]]
    ):
        raise RuntimeError("invalid HotUGPU reservation telemetry")
    return values


def validate_spatial_splits(
    split_texts: Mapping[str, str], *, expected_members: set[str]
) -> dict[str, tuple[str, ...]]:
    """Parse three pinned filename lists and require an exact partition."""
    if set(split_texts) != set(SPLIT_NAMES):
        raise ValueError(f"expected exactly these splits: {SPLIT_NAMES}")
    parsed: dict[str, tuple[str, ...]] = {}
    owners: dict[str, str] = {}
    for split in SPLIT_NAMES:
        rows = tuple(line.strip() for line in split_texts[split].splitlines() if line.strip())
        if not rows:
            raise ValueError(f"empty split: {split}")
        if any(
            PurePosixPath(row).name != row or "\\" in row or not row.lower().endswith(".jpg")
            for row in rows
        ):
            raise ValueError(f"invalid basename in {split}")
        if len(rows) != len(set(rows)):
            raise ValueError(f"duplicate filename in {split}")
        parsed[split] = rows
        for row in rows:
            if row in owners:
                raise ValueError(f"overlap: {row} appears in {owners[row]} and {split}")
            owners[row] = split
    missing = expected_members - set(owners)
    extra = set(owners) - expected_members
    if missing or extra:
        raise ValueError(f"split coverage mismatch: missing={len(missing)}, extra={len(extra)}")
    return parsed


def eurosat_class_from_filename(filename: str) -> str:
    """Return the class prefix from a canonical EuroSAT RGB filename."""
    path = PurePosixPath(filename)
    stem = path.stem
    label, separator, index = stem.rpartition("_")
    if (
        path.name != filename
        or "\\" in filename
        or path.suffix.lower() != ".jpg"
        or not label
        or not index.isdigit()
    ):
        raise ValueError(f"invalid EuroSAT filename: {filename}")
    return label


def require_probability_replay(saved, replayed, *, atol: float) -> float:
    """Require finite, shape-matched probability replay within an explicit tolerance."""
    if not np.isfinite(atol) or atol < 0:
        raise ValueError("probability replay tolerance must be finite and nonnegative")
    expected = np.asarray(saved, dtype=np.float64)
    actual = np.asarray(replayed, dtype=np.float64)
    if expected.shape != actual.shape or expected.ndim != 2:
        raise ValueError("probability replay shape mismatch")
    if not np.isfinite(expected).all() or not np.isfinite(actual).all():
        raise ValueError("probability replay values must be finite")
    delta = float(np.max(np.abs(expected - actual)))
    if delta > atol:
        raise ValueError(f"probability replay delta {delta} exceeds tolerance {atol}")
    return delta
