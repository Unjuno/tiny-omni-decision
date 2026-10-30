"""Integrity-checked paired scoring for fixed multimodal interventions."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence

import numpy as np

from .cluster_comparison import paired_cluster_interval


def _validate_probability_matrix(values: object, *, rows: int, options: int) -> np.ndarray:
    probabilities = np.asarray(values, dtype=np.float64)
    if (probabilities.shape != (rows, options) or not np.isfinite(probabilities).all()
            or (probabilities < 0).any() or (probabilities > 1).any()
            or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6, rtol=0)):
        raise ValueError("each intervention needs aligned normalized option probabilities")
    return probabilities


def _metrics(probabilities: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    predicted = probabilities.argmax(axis=1)
    correct = predicted == targets
    confidence = probabilities.max(axis=1)
    selected = probabilities[np.arange(len(targets)), targets]
    one_hot = np.eye(probabilities.shape[1], dtype=np.float64)[targets]
    bins = np.minimum((confidence * 15).astype(int), 14)
    ece = sum(
        abs(float((correct[bins == index] - confidence[bins == index]).sum()))
        for index in range(15)
    ) / len(targets)
    return {
        "accuracy": float(correct.mean()),
        "nll": float(-np.log(np.maximum(selected, 1e-12)).mean()),
        "brier": float(((probabilities - one_hot) ** 2).sum(axis=1).mean()),
        "ece_15": float(ece),
        "mean_confidence": float(confidence.mean()),
    }


def evaluate_intervention_factorial(
    *,
    evaluation_role: str,
    candidate_frozen: bool,
    frozen_candidate_sha256: str | None = None,
    sample_ids: Sequence[str],
    group_ids: Sequence[str],
    option_ids: Sequence[Sequence[str]],
    option_ids_by_arm: Mapping[str, Sequence[Sequence[str]]],
    intervention_types: Mapping[str, str],
    targets: Sequence[int],
    probabilities_by_arm: Mapping[str, object],
    baseline_arm: str = "matched",
) -> dict[str, object]:
    """Score saved predictions from matched modality interventions.

    ``probabilities_by_arm`` must already be realigned to each row's supplied
    option-ID order. This function never loads media or a model. It rejects
    sealed/test scoring unless the caller explicitly attests that the candidate
    is frozen; development comparisons remain paired by source group.
    """
    if not isinstance(evaluation_role, str) or evaluation_role not in {
        "train", "development", "validation", "sealed_audit", "test"
    }:
        raise ValueError("unsupported evaluation role")
    if type(candidate_frozen) is not bool:
        raise ValueError("candidate_frozen must be an explicit boolean")
    if evaluation_role in {"sealed_audit", "test"} and (
        candidate_frozen is not True
        or not isinstance(frozen_candidate_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", frozen_candidate_sha256) is None
    ):
        raise ValueError("sealed/test evaluation requires a frozen candidate SHA-256")
    if frozen_candidate_sha256 is not None and (
        not isinstance(frozen_candidate_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", frozen_candidate_sha256) is None
    ):
        raise ValueError("frozen candidate hash must be a lowercase SHA-256")
    if (isinstance(sample_ids, (str, bytes)) or isinstance(group_ids, (str, bytes))
            or isinstance(option_ids, (str, bytes)) or isinstance(targets, (str, bytes))):
        raise ValueError("sample, group, option and target inputs must be row sequences")
    ids, groups = list(sample_ids), list(group_ids)
    if any(isinstance(row, (str, bytes)) for row in option_ids):
        raise ValueError("option IDs must be a sequence of per-sample rows")
    candidates = [list(row) for row in option_ids]
    y = np.asarray(targets)
    count = len(ids)
    if (count == 0 or len(groups) != count or len(candidates) != count or y.shape != (count,)
            or y.dtype.kind not in "iu"):
        raise ValueError("nonempty aligned sample, group, option and target rows required")
    if (any(not isinstance(value, str) or not value.strip() for value in ids)
            or len(set(ids)) != count):
        raise ValueError("sample IDs must be nonempty and unique")
    if any(not isinstance(value, str) or not value.strip() for value in groups):
        raise ValueError("every sample needs a nonempty source group ID")
    if len(set(groups)) < 2:
        raise ValueError("paired group intervals require at least two independent groups")
    option_count = len(candidates[0])
    if option_count < 2 or any(len(row) != option_count for row in candidates):
        raise ValueError("this comparison requires a fixed option count of at least two")
    if any(any(not isinstance(item, str) or not item.strip() for item in row)
           or len(set(row)) != option_count for row in candidates):
        raise ValueError("each row needs unique nonempty option IDs")
    if (y < 0).any() or (y >= option_count).any():
        raise ValueError("targets must index the supplied option order")
    if baseline_arm not in probabilities_by_arm or len(probabilities_by_arm) < 2:
        raise ValueError("a matched baseline and at least one intervention arm are required")
    allowed_interventions = {
        "matched", "missing", "replace", "shuffle", "time_offset", "contradict",
        "option_permutation",
    }
    if (set(intervention_types) != set(probabilities_by_arm)
            or intervention_types.get(baseline_arm) != "matched"
            or any(not isinstance(value, str) or value not in allowed_interventions
                   for value in intervention_types.values())):
        raise ValueError("declare one supported intervention type per prediction arm")
    if set(option_ids_by_arm) != set(probabilities_by_arm):
        raise ValueError("each prediction arm needs its own aligned option IDs")
    for name, arm_options in option_ids_by_arm.items():
        if any(isinstance(row, (str, bytes)) for row in arm_options):
            raise ValueError("each arm must provide per-sample option ID rows")
        if [list(row) for row in arm_options] != candidates:
            raise ValueError(f"option IDs/order differ in intervention arm {name!r}")

    matrices = {
        name: _validate_probability_matrix(values, rows=count, options=option_count)
        for name, values in probabilities_by_arm.items()
    }
    metrics = {name: _metrics(values, y) for name, values in matrices.items()}
    baseline = matrices[baseline_arm]
    baseline_correct = (baseline.argmax(axis=1) == y).astype(np.float64)
    baseline_nll = -np.log(np.maximum(baseline[np.arange(count), y], 1e-12))
    baseline_brier = ((baseline - np.eye(option_count)[y]) ** 2).sum(axis=1)
    comparisons: dict[str, object] = {}
    for name, values in matrices.items():
        if name == baseline_arm:
            continue
        correct = (values.argmax(axis=1) == y).astype(np.float64)
        nll = -np.log(np.maximum(values[np.arange(count), y], 1e-12))
        brier = ((values - np.eye(option_count)[y]) ** 2).sum(axis=1)
        comparisons[name] = {
            "accuracy_delta": paired_cluster_interval(correct, baseline_correct, groups),
            "nll_delta": paired_cluster_interval(nll, baseline_nll, groups),
            "brier_delta": paired_cluster_interval(brier, baseline_brier, groups),
            "changed_prediction_count": int(
                (values.argmax(axis=1) != baseline.argmax(axis=1)).sum()
            ),
        }
    return {
        "schema_version": 1,
        "evaluation_role": evaluation_role,
        "candidate_frozen": candidate_frozen,
        "frozen_candidate_sha256": frozen_candidate_sha256,
        "baseline_arm": baseline_arm,
        "intervention_types": dict(intervention_types),
        "sample_count": count,
        "group_count": len(set(groups)),
        "sample_ids_sha256": _ordered_ids_hash(ids),
        "group_ids_sha256": _ordered_ids_hash(groups),
        "option_order_sha256": _ordered_ids_hash(
            ["\x1f".join(row) for row in candidates]
        ),
        "targets_sha256": _ordered_ids_hash([str(int(value)) for value in y]),
        "metrics": metrics,
        "paired_group_comparisons": comparisons,
        "uncertainty_note": (
            "Paired group intervals quantify sampling variation only; they do not remove "
            "checkpoint-selection, task-construction, or source-population bias."
        ),
    }


def _ordered_ids_hash(values: Sequence[str]) -> str:
    payload = json.dumps(list(values), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
