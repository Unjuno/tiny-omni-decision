from __future__ import annotations

import hashlib
import random
from collections import defaultdict

from .schema import DecisionExample


def _record_group(example: DecisionExample) -> tuple[str, str, str]:
    record_id = example.source_record_id
    # Typed Synth emits several question records from one state; keep the state intact.
    if example.source == "n4ze3m/typed-decisions-synth":
        record_id = record_id.split(":", 1)[0]
    return example.source, example.split, record_id


def partition_heldout_records(
    examples: list[DecisionExample],
    *,
    seed: int,
    validation_fraction: float = 0.5,
    heldout_partitions: dict[tuple[str, str], str] | None = None,
) -> tuple[list[DecisionExample], list[DecisionExample]]:
    """Keep declared upstream splits fixed and partition remaining records deterministically."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be strictly between zero and one")
    groups: dict[tuple[str, str, str], list[DecisionExample]] = defaultdict(list)
    for example in examples:
        groups[_record_group(example)].append(example)
    roles = heldout_partitions or {}
    source_splits = {(source, split) for source, split, _ in groups}
    if heldout_partitions is not None:
        missing = source_splits - roles.keys()
        if missing:
            raise ValueError(
                f"held-out source splits need explicit partition roles: {sorted(missing)}"
            )
    invalid = {role for role in roles.values()} - {"validation", "evaluation", "split"}
    if invalid:
        raise ValueError(f"unsupported held-out partition roles: {sorted(invalid)}")

    groups_by_source_split: dict[tuple[str, str], list[tuple[str, str, str]]] = defaultdict(list)
    for group in groups:
        groups_by_source_split[group[:2]].append(group)

    validation_groups: set[tuple[str, str, str]] = set()
    evaluation_groups: set[tuple[str, str, str]] = set()
    for source_split, group_ids in sorted(groups_by_source_split.items()):
        source, split = source_split
        role = roles.get(source_split, "split")
        if role == "validation":
            validation_groups.update(group_ids)
            continue
        if role == "evaluation":
            evaluation_groups.update(group_ids)
            continue
        ordered = sorted(group_ids)
        random.Random(f"{seed}:{source}:{split}:heldout-partition").shuffle(ordered)
        if len(ordered) < 2:
            raise ValueError(
                f"source {source} split {split} needs at least two held-out records to split"
            )
        validation_count = min(
            len(ordered) - 1, max(1, round(len(ordered) * validation_fraction))
        )
        validation_groups.update(ordered[:validation_count])
        evaluation_groups.update(ordered[validation_count:])

    validation = [example for example in examples if _record_group(example) in validation_groups]
    evaluation = [example for example in examples if _record_group(example) in evaluation_groups]
    return validation, evaluation


def comparison_deltas(
    baseline: dict[str, dict[str, float | int]],
    trained: dict[str, dict[str, float | int]],
) -> dict[str, dict[str, float]]:
    """Return trained-minus-base metric changes for shared overall/group keys."""
    delta: dict[str, dict[str, float]] = {}
    for group in sorted(baseline.keys() & trained.keys()):
        delta[group] = {
            metric: float(trained[group][metric]) - float(baseline[group][metric])
            for metric in ("accuracy", "nll", "brier", "ece", "mean_confidence")
        }
    return delta


def selection_loss(metrics: dict[str, float | int], *, brier_weight: float = 0.2) -> float:
    return float(metrics["nll"]) + brier_weight * float(metrics["brier"])


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
