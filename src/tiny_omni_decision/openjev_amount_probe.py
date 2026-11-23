"""Train-only Open-Jev amount-control helpers for frozen text encoders."""

from __future__ import annotations

import hashlib
import json
import random
import re
import unicodedata
from collections import defaultdict
from math import log
from typing import Any

ALLOWED_STATE_FIELDS = frozenset({"text", "policy", "requested_role"})
FORBIDDEN_STATE_FIELDS = frozenset({"selected", "selected_id", "candidates"})


def parse_object(value: Any, *, field: str) -> dict[str, Any]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise ValueError(f"{field} must decode to an object")
    return value


def sanitize_state(value: Any) -> dict[str, Any]:
    """Return only source-visible inputs; never expose candidate/answer metadata."""
    state = parse_object(value, field="state")
    if not isinstance(state.get("text"), str) or not state["text"].strip():
        raise ValueError("state.text must be a non-empty string")
    clean = {key: state[key] for key in sorted(ALLOWED_STATE_FIELDS) if key in state}
    # Round-trip through JSON to reject non-serializable values before tokenization.
    json.dumps(clean, sort_keys=True, ensure_ascii=False)
    if FORBIDDEN_STATE_FIELDS.intersection(clean):
        raise AssertionError("forbidden state field reached the model input")
    return clean


def normalized_content_hash(value: Any) -> str:
    serialized = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    normalized = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", serialized).casefold()).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def make_family_split(
    rows: list[dict[str, Any]], *, seed: int = 17, validation_fraction: float = 0.2
) -> dict[str, str]:
    """Deterministically hold out whole document families from the publisher train split."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    families: set[str] = set()
    for row in rows:
        metadata = parse_object(row.get("metadata_json") or "{}", field="metadata_json")
        family = str(metadata.get("document_family_id", "")).strip()
        if not family:
            raise ValueError("every train row must have document_family_id")
        families.add(family)
    ordered = sorted(families)
    if len(ordered) < 5:
        raise ValueError("at least five document families are required")
    random.Random(seed).shuffle(ordered)
    validation_count = max(1, round(len(ordered) * validation_fraction))
    if validation_count >= len(ordered):
        raise ValueError("split would leave no training families")
    held_out = set(ordered[:validation_count])
    return {family: ("validation" if family in held_out else "train") for family in families}


def validate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize rows without carrying privileged state or metadata into model text."""
    normalized: list[dict[str, Any]] = []
    ids: set[str] = set()
    for row in rows:
        sample_id = str(row.get("id", "")).strip()
        if not sample_id or sample_id in ids:
            raise ValueError("sample IDs must be non-empty and unique")
        ids.add(sample_id)
        options = row.get("options")
        target = row.get("target")
        if isinstance(options, str):
            options = json.loads(options)
        if isinstance(target, str):
            target = json.loads(target)
        if not isinstance(options, list) or len(options) < 2:
            raise ValueError(f"{sample_id}: at least two options are required")
        if not isinstance(target, list) or len(target) != len(options):
            raise ValueError(f"{sample_id}: target length must match options")
        probabilities = [float(value) for value in target]
        if any(value < 0.0 for value in probabilities) or abs(sum(probabilities) - 1.0) > 1e-5:
            raise ValueError(f"{sample_id}: target must be a probability distribution")
        state = sanitize_state(row.get("state_json") or "{}")
        metadata = parse_object(row.get("metadata_json") or "{}", field="metadata_json")
        normalized.append(
            {
                "id": sample_id,
                "family": str(metadata["document_family_id"]),
                "question_id": str(metadata.get("question_id", "UNKNOWN")),
                "kind": str(row.get("kind", "UNKNOWN")),
                "question": str(row.get("question", "")),
                "state": state,
                "options": [str(option) for option in options],
                "target": probabilities,
                "input_hash": normalized_content_hash(
                    [state, row.get("question"), row.get("kind"), options]
                ),
            }
        )
    return normalized


def assert_split_integrity(rows: list[dict[str, Any]], split_by_family: dict[str, str]) -> None:
    """Fail closed on family, sample-ID, or normalized-content overlap."""
    seen_ids: dict[str, str] = {}
    seen_inputs: dict[str, str] = {}
    for row in rows:
        family = row["family"]
        split = split_by_family.get(family)
        if split not in {"train", "validation"}:
            raise ValueError(f"family {family!r} has no valid split assignment")
        for key, seen in (("id", seen_ids), ("input_hash", seen_inputs)):
            value = row[key]
            previous = seen.setdefault(value, split)
            if previous != split:
                raise ValueError(f"{key} overlap crosses train/validation")


def format_pair(row: dict[str, Any], option: str) -> tuple[str, str]:
    """Format one supplied-option decision without leaking target or metadata."""
    state_text = json.dumps(row["state"], ensure_ascii=False, sort_keys=True)
    first = f"Visible state: {state_text}"
    second = f"Question: {row['question']}\nDecision type: {row['kind']}\nOption: {option}"
    return first, second


def question_type_cells(records: list[dict[str, Any]]) -> dict[str, list[int]]:
    cells: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(records):
        cells[row["question_id"]].append(index)
    return dict(sorted(cells.items()))


def _prediction_losses(record: dict[str, Any]) -> dict[str, float]:
    target = [float(value) for value in record["target"]]
    probabilities = [float(value) for value in record["probabilities"]]
    if len(target) != len(probabilities) or not target:
        raise ValueError("prediction and target vectors must have equal non-zero lengths")
    if any(value < 0.0 for value in target) or abs(sum(target) - 1.0) > 1e-5:
        raise ValueError("target must be a probability distribution")
    if any(value < 0.0 or value > 1.0 for value in probabilities):
        raise ValueError("prediction probabilities must be within [0, 1]")
    if abs(sum(probabilities) - 1.0) > 1e-5:
        raise ValueError("prediction probabilities must sum to one")
    target_index = max(range(len(target)), key=target.__getitem__)
    prediction_index = max(range(len(probabilities)), key=probabilities.__getitem__)
    return {
        "accuracy": float(prediction_index == target_index),
        "nll": -sum(target[i] * log(max(probabilities[i], 1e-12)) for i in range(len(target))),
        "brier": sum((probabilities[i] - target[i]) ** 2 for i in range(len(target))),
    }


def paired_family_bootstrap(
    reference: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    *,
    seed: int = 17,
    iterations: int = 2000,
) -> dict[str, Any]:
    """Paired percentile bootstrap, resampling whole validation families.

    Deltas are candidate minus reference. Macro metrics average question_id cells,
    and each cell averages examples within the resampled families.
    """
    if iterations < 100:
        raise ValueError("at least 100 bootstrap iterations are required")
    reference_by_id = {str(row["id"]): row for row in reference}
    candidate_by_id = {str(row["id"]): row for row in candidate}
    if len(reference_by_id) != len(reference) or len(candidate_by_id) != len(candidate):
        raise ValueError("prediction IDs must be unique")
    if reference_by_id.keys() != candidate_by_id.keys():
        raise ValueError("paired models must have identical prediction IDs")
    paired = []
    for sample_id in sorted(reference_by_id):
        left, right = reference_by_id[sample_id], candidate_by_id[sample_id]
        for key in ("family", "question_id", "target"):
            if left[key] != right[key]:
                raise ValueError(f"paired predictions disagree on {key} for {sample_id}")
        left_metrics = _prediction_losses(left)
        right_metrics = _prediction_losses(right)
        paired.append(
            {
                "family": str(left["family"]),
                "question_id": str(left["question_id"]),
                "delta": {
                    key: right_metrics[key] - left_metrics[key]
                    for key in ("accuracy", "nll", "brier")
                },
            }
        )
    families = sorted({row["family"] for row in paired})
    if len(families) < 2:
        raise ValueError("at least two validation families are required")
    rows_by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in paired:
        rows_by_family[row["family"]].append(row)

    def macro_delta(selected_families: list[str]) -> dict[str, float]:
        counts: dict[str, int] = defaultdict(int)
        sums: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        for family in selected_families:
            for row in rows_by_family[family]:
                question_id = row["question_id"]
                counts[question_id] += 1
                for metric, value in row["delta"].items():
                    sums[metric][question_id] += value
        present = sorted(counts)
        if not present:
            raise ValueError("bootstrap sample contained no validation rows")
        return {
            metric: sum(sums[metric][question_id] / counts[question_id] for question_id in present)
            / len(present)
            for metric in ("accuracy", "nll", "brier")
        }

    point = macro_delta(families)
    rng = random.Random(seed)
    samples = {metric: [] for metric in point}
    for _ in range(iterations):
        selected = [rng.choice(families) for _ in families]
        delta = macro_delta(selected)
        for metric, value in delta.items():
            samples[metric].append(value)
    summary = {}
    for metric, values in samples.items():
        summary[metric] = {
            "candidate_minus_reference": point[metric],
            "percentile_95_ci": [float(value) for value in np_percentile(values, [2.5, 97.5])],
            "bootstrap_probability_candidate_better": float(
                sum(value > 0 for value in values) / iterations
                if metric == "accuracy"
                else sum(value < 0 for value in values) / iterations
            ),
        }
    return {
        "reference_minus_candidate_order": "candidate minus reference",
        "resampling_unit": "document_family_id",
        "validation_families": len(families),
        "validation_predictions": len(paired),
        "iterations": iterations,
        "seed": seed,
        "metrics": summary,
        "inference_limit": "Internal validation selected checkpoints; intervals are exploratory "
        "and conditional on selection.",
    }


def np_percentile(values: list[float], percentiles: list[float]) -> list[float]:
    """Small NumPy-free linear percentile helper for bootstrap summaries."""
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot compute a percentile of an empty sample")
    result = []
    for percentile in percentiles:
        location = (len(ordered) - 1) * percentile / 100.0
        lower = int(location)
        upper = min(lower + 1, len(ordered) - 1)
        fraction = location - lower
        result.append(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)
    return result
