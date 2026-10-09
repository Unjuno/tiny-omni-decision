"""Export pairing and manifest helpers for frozen EmbeddingGemma student bundles."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def align_teacher_predictions(
    examples: Sequence[Any], teacher_predictions: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Pair Teacher option probabilities with exact corpus choices by immutable sample ID."""
    by_id: dict[str, Mapping[str, Any]] = {}
    for row in teacher_predictions:
        sample_id = str(row.get("sample_id", ""))
        if not sample_id or sample_id in by_id:
            raise ValueError("Teacher prediction sample IDs must be non-empty and unique")
        by_id[sample_id] = row

    expected_ids = [example.id for example in examples]
    if len(set(expected_ids)) != len(expected_ids):
        raise ValueError("evaluation example IDs must be unique")
    if set(expected_ids) != set(by_id):
        raise ValueError("Teacher predictions and evaluation examples have different IDs")

    aligned: list[dict[str, Any]] = []
    for example in examples:
        row = by_id[example.id]
        probabilities = row.get("option_probabilities")
        expected_target = example.options.index(example.target)
        expected_labels = [chr(ord("A") + index) for index in range(len(example.options))]
        if (
            row.get("modality") != example.modality
            or row.get("source") != example.source
            or row.get("target") != expected_target
            or row.get("option_labels") != expected_labels
            or not isinstance(probabilities, list)
            or len(probabilities) != len(example.options)
        ):
            raise ValueError(f"Teacher choices/order metadata mismatch for {example.id}")
        aligned.append(
            {
                "sample_id": example.id,
                "source": example.source,
                "modality": example.modality,
                "target": expected_target,
                "options": list(example.options),
                "option_probabilities": [float(value) for value in probabilities],
            }
        )
    return aligned
