"""Deterministic, storm-grouped sub-SEVIR forecast window construction."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ForecastWindow:
    """Eight historical frames and the immediately following target frame."""

    event_id: str
    split: str
    input_indices: tuple[int, ...]
    target_index: int
    target: int

    @property
    def sample_id(self) -> str:
        return f"{self.event_id}:target-{self.target_index}"


def event_split(event_id: str, *, seed: int = 17, validation_fraction: float = 0.2) -> str:
    """Assign a group deterministically without consulting labels."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be strictly between 0 and 1")
    digest = hashlib.sha256(f"subsevir-event-split-v1\0{seed}\0{event_id}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    return "validation" if value < validation_fraction else "train"


def build_forecast_windows(
    event_ids: Sequence[str],
    timestamps_minutes: Sequence[int],
    labels: Sequence[int],
    *,
    seed: int = 17,
    validation_fraction: float = 0.2,
    history_frames: int = 8,
    cadence_minutes: int = 5,
) -> list[ForecastWindow]:
    """Build eligible windows, keeping every storm event wholly in one split.

    A window is eligible only when all history frames and its target form a
    contiguous sequence at the declared cadence. The target is the source
    ``label_1d_class`` at the next frame. No label is used in split assignment.
    """
    if not (len(event_ids) == len(timestamps_minutes) == len(labels)):
        raise ValueError("event_ids, timestamps_minutes and labels must have equal lengths")
    if history_frames < 1 or cadence_minutes < 1:
        raise ValueError("history_frames and cadence_minutes must be positive")

    grouped: dict[str, list[int]] = defaultdict(list)
    for index, event_id in enumerate(event_ids):
        grouped[str(event_id)].append(index)

    windows: list[ForecastWindow] = []
    for event_id, indices in grouped.items():
        if indices != list(range(indices[0], indices[-1] + 1)):
            raise ValueError(f"rows for event {event_id!r} are not contiguous")
        split = event_split(event_id, seed=seed, validation_fraction=validation_fraction)
        for offset in range(history_frames, len(indices)):
            sequence = indices[offset - history_frames : offset + 1]
            cadence = [
                int(timestamps_minutes[right]) - int(timestamps_minutes[left])
                for left, right in zip(sequence, sequence[1:], strict=False)
            ]
            if any(delta != cadence_minutes for delta in cadence):
                continue
            target_index = indices[offset]
            target = int(labels[target_index])
            if target not in (0, 1):
                raise ValueError(f"non-binary target at row {target_index}: {target}")
            windows.append(
                ForecastWindow(
                    event_id=event_id,
                    split=split,
                    input_indices=tuple(indices[offset - history_frames : offset]),
                    target_index=target_index,
                    target=target,
                )
            )
    return sorted(windows, key=lambda item: (item.split, item.event_id, item.target_index))


def window_dict(window: ForecastWindow) -> dict[str, object]:
    """JSON-friendly representation used by the frozen external manifest."""
    result = asdict(window)
    result["sample_id"] = window.sample_id
    return result


def canonical_sha256(records: Iterable[object]) -> str:
    """Hash a deterministic JSON-compatible sequence."""
    import json

    payload = json.dumps(list(records), ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
