"""Deterministic, storm-grouped sub-SEVIR forecast window construction."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
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


def fresh_training_event_split(
    event_id: str, *, seed: int = 17, validation_fraction: float = 0.2
) -> str:
    """Split only prior training events for a new, independent development fit.

    The namespace intentionally differs from :func:`event_split`, so the new
    candidate split cannot silently reuse the original train/validation map.
    Callers must first exclude every event from the source manifest's
    validation partition.
    """
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be strictly between 0 and 1")
    digest = hashlib.sha256(
        f"subsevir-gated-fresh-train-split-v1\0{seed}\0{event_id}".encode()
    ).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    return "validation" if value < validation_fraction else "train"


def build_fresh_training_manifest(
    source_manifest: Mapping[str, object],
    *,
    parent_manifest_sha256: str,
    seed: int = 17,
    validation_fraction: float = 0.2,
) -> dict[str, object]:
    """Create a new split from source-train events, excluding source validation.

    Every window in the new validation partition is a previously untouched
    validation example for this candidate fit. The source manifest's existing
    validation groups are not copied into the result.
    """
    try:
        split_group_ids = source_manifest["split_group_ids"]
        windows = source_manifest["windows"]
    except KeyError as exc:
        raise ValueError("source manifest lacks split groups or windows") from exc
    if not isinstance(split_group_ids, Mapping) or not isinstance(windows, list):
        raise ValueError("source manifest split groups/windows have invalid types")
    source_train_ids = {str(value) for value in split_group_ids.get("train", [])}
    source_validation_ids = {
        str(value) for value in split_group_ids.get("validation", [])
    }
    if not source_train_ids or not source_validation_ids:
        raise ValueError("source manifest must have non-empty train and validation groups")
    if source_train_ids & source_validation_ids:
        raise ValueError("source manifest has event overlap between splits")

    selected_windows: list[dict[str, object]] = []
    new_event_splits: dict[str, str] = {}
    for raw_window in windows:
        if not isinstance(raw_window, Mapping):
            raise ValueError("source window must be a mapping")
        source_split = str(raw_window.get("split"))
        event_id = str(raw_window.get("event_id"))
        if source_split == "validation":
            if event_id not in source_validation_ids:
                raise ValueError(f"source validation event is unregistered: {event_id}")
            continue
        if source_split != "train" or event_id not in source_train_ids:
            raise ValueError(f"window is outside the source training partition: {event_id}")
        split = fresh_training_event_split(
            event_id, seed=seed, validation_fraction=validation_fraction
        )
        previous = new_event_splits.setdefault(event_id, split)
        if previous != split:
            raise ValueError(f"event crosses new splits: {event_id}")
        window = dict(raw_window)
        window["split"] = split
        selected_windows.append(window)

    if set(new_event_splits) != source_train_ids:
        missing = sorted(source_train_ids - set(new_event_splits))
        raise ValueError(f"source training groups have no eligible windows: {missing[:5]}")
    selected_windows.sort(
        key=lambda item: (str(item["split"]), str(item["event_id"]), int(item["target_index"]))
    )
    groups = {
        split: sorted(
            event_id for event_id, assigned_split in new_event_splits.items()
            if assigned_split == split
        )
        for split in ("train", "validation")
    }
    if not groups["train"] or not groups["validation"]:
        raise ValueError("fresh split produced an empty partition")
    if set(groups["train"]) & set(groups["validation"]):
        raise AssertionError("fresh split contains event overlap")
    grouped_windows = {
        split: [window for window in selected_windows if window["split"] == split]
        for split in ("train", "validation")
    }
    if any(not grouped_windows[split] for split in grouped_windows):
        raise ValueError("fresh split produced a partition without windows")
    target_counts = {
        split: dict(sorted(Counter(int(window["target"]) for window in items).items()))
        for split, items in grouped_windows.items()
    }
    manifest: dict[str, object] = {
        "schema": "tiny-omni-decision/subsevir-gated-fresh-split-manifest-v1",
        "parent_manifest_sha256": parent_manifest_sha256,
        "seed": seed,
        "validation_fraction": validation_fraction,
        "split_method": (
            "SHA-256(namespace, seed, event_id) over source-manifest train groups only; "
            "label-independent; source-manifest validation groups excluded"
        ),
        "source_train_group_count": len(source_train_ids),
        "source_validation_group_count_excluded": len(source_validation_ids),
        "split_group_ids": groups,
        "split_group_counts": {split: len(items) for split, items in groups.items()},
        "split_window_counts": {
            split: len(items) for split, items in grouped_windows.items()
        },
        "split_target_counts": target_counts,
        "windows_sha256": canonical_sha256(selected_windows),
        "windows": selected_windows,
    }
    return manifest


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
