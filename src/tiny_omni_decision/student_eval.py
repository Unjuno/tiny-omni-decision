"""Validation evaluation for the supplied-option embedding student."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from .decision_math import brier_score, expected_calibration_error, negative_log_likelihood
from .schema import DecisionExample
from .student import (
    model_sentence_embeddings,
    processor_inputs_for_decision_example,
    processor_inputs_for_options,
    supplied_option_logits,
)


def _measure(rows: list[tuple[list[float], int]], *, ece_bins: int) -> dict[str, float | int]:
    if not rows:
        raise ValueError("cannot measure an empty prediction group")
    probabilities = [row[0] for row in rows]
    targets = [row[1] for row in rows]
    return {
        "count": len(rows),
        "accuracy": sum(
            max(range(len(values)), key=values.__getitem__) == target
            for values, target in rows
        )
        / len(rows),
        "nll": negative_log_likelihood(probabilities, targets),
        "brier": brier_score(probabilities, targets),
        "ece": expected_calibration_error(probabilities, targets, ece_bins),
        "mean_confidence": sum(max(row) for row in probabilities) / len(probabilities),
    }


def evaluate_student_examples(
    model: Any,
    processor: Any,
    examples: Iterable[DecisionExample],
    *,
    data_root: Any,
    temperature: float,
    ece_bins: int = 15,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate examples in input order and report modality/source calibration metrics."""
    import math

    import torch

    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if ece_bins < 1:
        raise ValueError("ece_bins must be positive")

    materialized = list(examples)
    if not materialized:
        raise ValueError("examples must be non-empty")
    if len({example.id for example in materialized}) != len(materialized):
        raise ValueError("example IDs must be unique")

    model.eval()
    groups: dict[str, list[tuple[list[float], int]]] = defaultdict(list)
    predictions: list[dict[str, Any]] = []
    with torch.inference_mode():
        for example in materialized:
            query_inputs = processor_inputs_for_decision_example(
                processor, example, data_root=data_root
            )
            option_inputs = processor_inputs_for_options(processor, example.options)
            query_embedding = model_sentence_embeddings(model, query_inputs)[0]
            option_embeddings = model_sentence_embeddings(model, option_inputs)
            logits = supplied_option_logits(
                query_embedding, option_embeddings, temperature=temperature
            )
            probabilities = torch.softmax(logits.float(), dim=-1).cpu().tolist()
            target = example.options.index(example.target)
            option_order_hash = hashlib.sha256(
                json.dumps(example.options, ensure_ascii=False, separators=(",", ":")).encode(
                    "utf-8"
                )
            ).hexdigest()
            record = {
                "sample_id": example.id,
                "source": example.source,
                "modality": example.modality,
                "target": target,
                "options": list(example.options),
                "option_order_sha256": option_order_hash,
                "option_logits": logits.float().cpu().tolist(),
                "option_probabilities": probabilities,
                "prediction": int(max(range(len(probabilities)), key=probabilities.__getitem__)),
                "confidence": float(max(probabilities)),
            }
            predictions.append(record)
            groups["all"].append((probabilities, target))
            groups[f"modality:{example.modality}"].append((probabilities, target))
            groups[f"source:{example.source}"].append((probabilities, target))

    metrics = {
        key: _measure(rows, ece_bins=ece_bins) for key, rows in sorted(groups.items())
    }
    modality_metrics = [
        value for key, value in metrics.items() if key.startswith("modality:")
    ]
    metrics["macro_modality"] = {
        name: sum(float(row[name]) for row in modality_metrics) / len(modality_metrics)
        for name in ("accuracy", "nll", "brier", "ece", "mean_confidence")
    }
    metrics["macro_modality"]["count"] = len(modality_metrics)
    metrics["minimum_modality_accuracy"] = min(
        float(row["accuracy"]) for row in modality_metrics
    )
    return metrics, predictions
