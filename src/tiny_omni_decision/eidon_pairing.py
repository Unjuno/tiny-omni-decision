"""Utilities for contributor-safe Eidon video-to-sensor matching probes."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Mapping, Sequence

import numpy as np

from tiny_omni_decision.quaternion_features import quaternion_rotation_matrices


def build_four_choice_queries(
    records: Sequence[Mapping[str, object]], *, seed: int = 17
) -> list[dict[str, object]]:
    """Build exact-pair four-choice queries from one contributor/task group.

    Each group must contain exactly four distinct recordings. Candidate order is
    deterministic for a seed and independently shuffled for each query.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if not records:
        raise ValueError("records must be non-empty")

    required = {"recording_id", "contributor_id", "task_type", "split", "asset_sha256"}
    groups: dict[tuple[object, object, object], list[Mapping[str, object]]] = defaultdict(list)
    contributor_splits: dict[object, object] = {}
    asset_splits: dict[str, object] = {}
    recording_ids: set[object] = set()

    for row in records:
        missing = required - row.keys()
        if missing:
            raise ValueError(f"record is missing required fields: {sorted(missing)}")
        recording_id = row["recording_id"]
        contributor_id = row["contributor_id"]
        task_type = row["task_type"]
        split = row["split"]
        asset_hash = row["asset_sha256"]
        if any(value is None or value == "" for value in (recording_id, contributor_id,
                                                               task_type, split, asset_hash)):
            raise ValueError("record identity, group, split, and asset hash must be non-empty")
        try:
            hash(recording_id)
            hash(contributor_id)
        except TypeError as exc:
            raise ValueError("recording and contributor IDs must be hashable") from exc
        if recording_id in recording_ids:
            raise ValueError("recording IDs must be unique")
        recording_ids.add(recording_id)

        previous_split = contributor_splits.setdefault(contributor_id, split)
        if previous_split != split:
            raise ValueError("contributor appears in multiple splits")
        previous_asset_split = asset_splits.setdefault(str(asset_hash), split)
        if previous_asset_split != split:
            raise ValueError("asset hash appears in multiple splits")
        groups[(split, contributor_id, task_type)].append(row)

    queries = []
    for (split, contributor_id, task_type), members in sorted(
        groups.items(), key=lambda item: tuple(str(value) for value in item[0])
    ):
        if len(members) != 4 or len({member["recording_id"] for member in members}) != 4:
            raise ValueError(
                "each contributor/task group must contain exactly four unique recordings"
            )
        if len({str(member["asset_sha256"]) for member in members}) != 4:
            raise ValueError("each contributor/task group must contain four unique media assets")
        candidate_ids = sorted((member["recording_id"] for member in members), key=str)
        for query_id in candidate_ids:
            options = list(candidate_ids)
            query_seed = f"{seed}:{split}:{contributor_id}:{task_type}:{query_id}"
            random.Random(query_seed).shuffle(options)
            queries.append({
                "split": split,
                "contributor_id": contributor_id,
                "task_type": task_type,
                "query_recording_id": query_id,
                "options": options,
                "target_index": options.index(query_id),
            })
    return queries


def relative_orientation_window_features(
    timestamps_ms: Sequence[float],
    slots: Sequence[int],
    quaternions_xyzw: Sequence[Sequence[float]],
    *,
    window_start_ms: float,
    window_end_ms: float,
    bins: int = 8,
) -> np.ndarray:
    """Return chest-relative rotation-matrix means for six limbs over time bins.

    Inputs contain one row per `(time_ms, slot)` and XYZW unit quaternions for
    slots 0..6. Slot 6 is the chest reference. Incomplete timestamps and empty
    bins fail closed; absent samples are never interpolated.
    """
    times = np.asarray(timestamps_ms, dtype=np.float64)
    slot_ids = np.asarray(slots)
    quats = np.asarray(quaternions_xyzw, dtype=np.float64)
    if (times.ndim != 1 or slot_ids.ndim != 1 or quats.ndim != 2
            or quats.shape != (len(times), 4) or len(slot_ids) != len(times)
            or len(times) == 0 or not np.isfinite(times).all()
            or not np.isfinite(quats).all()):
        raise ValueError("finite time, slot, and XYZW quaternion rows are required")
    if (isinstance(bins, bool) or not isinstance(bins, int) or bins < 1
            or not np.isfinite(window_start_ms) or not np.isfinite(window_end_ms)
            or window_end_ms <= window_start_ms):
        raise ValueError("a positive bin count and finite increasing window are required")
    if not np.issubdtype(slot_ids.dtype, np.integer) or np.any((slot_ids < 0) | (slot_ids > 6)):
        raise ValueError("sensor slots must be integers from 0 through 6")
    slot_ids = slot_ids.astype(np.int64, copy=False)

    selected = (times >= window_start_ms) & (times < window_end_ms)
    times, slot_ids, quats = times[selected], slot_ids[selected], quats[selected]
    if len(times) == 0:
        raise ValueError("sensor window contains no samples")
    pairs = np.column_stack((times, slot_ids))
    if len(np.unique(pairs, axis=0)) != len(pairs):
        raise ValueError("duplicate time/slot sensor rows are not allowed")

    unique_times, inverse = np.unique(times, return_inverse=True)
    quats_by_slot = np.zeros((len(unique_times), 7, 4), dtype=np.float64)
    present = np.zeros((len(unique_times), 7), dtype=np.bool_)
    quats_by_slot[inverse, slot_ids] = quats
    present[inverse, slot_ids] = True
    if not present.all():
        raise ValueError("every timestamp must contain slots 0 through 6")

    rotations = quaternion_rotation_matrices(
        quats_by_slot[..., [3, 0, 1, 2]]
    ).reshape(len(unique_times), 7, 3, 3)
    chest = rotations[:, 6]
    relative = np.einsum("tji,tsjk->tsik", chest, rotations[:, :6])
    values = relative.reshape(len(unique_times), 54)
    bin_ids = np.floor(
        (unique_times - window_start_ms) / (window_end_ms - window_start_ms) * bins
    ).astype(np.int64)
    bin_ids = np.clip(bin_ids, 0, bins - 1)
    counts = np.bincount(bin_ids, minlength=bins)
    if np.any(counts == 0):
        raise ValueError("every temporal bin must contain at least one sensor timestamp")
    features = np.stack([values[bin_ids == index].mean(axis=0) for index in range(bins)])
    if not np.isfinite(features).all():
        raise ValueError("relative orientation features must be finite")
    return features.astype(np.float32)
