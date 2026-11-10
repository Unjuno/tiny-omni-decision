"""Deterministic, source-grouped folds for the RoboMNIST AV readout CV."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any


def make_pair_folds(
    pair_units: list[dict[str, Any]], *, fold_count: int = 5, seed: int = 17
) -> dict[str, int]:
    """Assign whole pair units to balanced, deterministic folds.

    Each source group must belong to exactly one pair unit; otherwise folds
    could leak an audio/video asset through a different pair.
    """
    if fold_count < 2 or seed < 0:
        raise ValueError("fold_count must be >= 2 and seed must be non-negative")
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_pair_ids: set[str] = set()
    source_owner: dict[str, str] = {}
    for unit in pair_units:
        pair_id = unit.get("pair_unit_id")
        condition = unit.get("condition")
        group_a, group_b = unit.get("group_a"), unit.get("group_b")
        if (
            not isinstance(pair_id, str)
            or not pair_id
            or pair_id in seen_pair_ids
            or not isinstance(condition, str)
            or not condition
            or not isinstance(group_a, str)
            or not isinstance(group_b, str)
            or not group_a
            or not group_b
            or group_a == group_b
        ):
            raise ValueError("pair units must have unique IDs and two distinct source groups")
        seen_pair_ids.add(pair_id)
        for source_group in (group_a, group_b):
            previous = source_owner.setdefault(source_group, pair_id)
            if previous != pair_id:
                raise ValueError("a source group is reused across pair units")
        by_condition[condition].append(unit)
    if len(seen_pair_ids) < fold_count:
        raise ValueError("not enough pair units for the requested folds")

    assignment: dict[str, int] = {}
    fold_loads = [0] * fold_count
    ordered_conditions = sorted(by_condition.items(), key=lambda item: (-len(item[1]), item[0]))
    for condition, units in ordered_conditions:
        units = sorted(
            units,
            key=lambda unit: (
                hashlib.sha256(
                    f"{seed}\0{condition}\0{unit['pair_unit_id']}".encode()
                ).digest(),
                unit["pair_unit_id"],
            ),
        )
        offsets = []
        for offset in range(fold_count):
            additions = [0] * fold_count
            for index in range(len(units)):
                additions[(index + offset) % fold_count] += 1
            loads = [
                current + addition
                for current, addition in zip(fold_loads, additions, strict=True)
            ]
            squared_imbalance = sum(
                (load * fold_count - len(seen_pair_ids)) ** 2 for load in loads
            )
            offsets.append((max(loads), squared_imbalance, offset, additions))
        _, _, offset, additions = min(offsets, key=lambda item: item[:3])
        for index, unit in enumerate(units):
            assignment[unit["pair_unit_id"]] = (index + offset) % fold_count
        fold_loads = [
            current + addition
            for current, addition in zip(fold_loads, additions, strict=True)
        ]
    return assignment


def select_capacity_rank(
    metrics_by_rank: dict[int, dict[str, float]],
    *,
    candidate_rank: int = 8,
    reference_rank: int = 32,
    min_nll_gain: float = 0.02,
    max_accuracy_loss: float = 0.01,
) -> int:
    """Select the smaller rank only for a predeclared OOF-quality gain."""
    if candidate_rank not in metrics_by_rank or reference_rank not in metrics_by_rank:
        raise ValueError("both candidate and reference ranks must be evaluated")
    candidate = metrics_by_rank[candidate_rank]
    reference = metrics_by_rank[reference_rank]
    for metrics in (candidate, reference):
        if not all(key in metrics for key in ("nll", "accuracy")):
            raise ValueError("capacity metrics require nll and accuracy")
        if not all(float(metrics[key]) == float(metrics[key]) for key in ("nll", "accuracy")):
            raise ValueError("capacity metrics must be finite")
    if (
        reference["nll"] - candidate["nll"] >= min_nll_gain
        and candidate["accuracy"] >= reference["accuracy"] - max_accuracy_loss
    ):
        return candidate_rank
    return reference_rank
