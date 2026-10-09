"""Validation evaluation for the supplied-option embedding student."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from .decision_math import brier_score, expected_calibration_error, negative_log_likelihood
from .schema import DecisionExample
from .student import (
    decision_option_text,
    model_sentence_embeddings,
    processor_inputs_for_decision_example,
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


def _read_jsonl(path: Path) -> tuple[bytes, list[dict[str, Any]]]:
    raw = path.read_bytes()
    rows = [json.loads(line) for line in raw.splitlines() if line]
    return raw, rows


def load_fixed_validation_snapshot(
    snapshot_path: str | Path,
    selection_manifest_path: str | Path,
    *,
    source_validation_path: str | Path,
    teacher_predictions_path: str | Path,
) -> tuple[list[DecisionExample], list[dict[str, Any]]]:
    """Verify the frozen validation subset against its corpus and Teacher prediction order."""
    manifest = json.loads(Path(selection_manifest_path).read_text(encoding="utf-8"))
    if manifest.get("split") != "validation":
        raise ValueError("selection manifest must identify the validation split")
    snapshot_bytes, selected_rows = _read_jsonl(Path(snapshot_path))
    source_bytes, source_rows = _read_jsonl(Path(source_validation_path))
    teacher_bytes, teacher_rows = _read_jsonl(Path(teacher_predictions_path))
    checks = (
        (hashlib.sha256(snapshot_bytes).hexdigest(), manifest["selected_validation_jsonl_sha256"]),
        (hashlib.sha256(source_bytes).hexdigest(), manifest["source_validation_jsonl_sha256"]),
        (
            hashlib.sha256(teacher_bytes).hexdigest(),
            manifest["teacher_best_validation_predictions_sha256"],
        ),
    )
    if any(actual != expected for actual, expected in checks):
        raise ValueError("validation snapshot, source corpus, or Teacher prediction hash mismatch")

    selected_ids = [row["id"] for row in selected_rows]
    teacher_ids = [row["sample_id"] for row in teacher_rows]
    if selected_ids != manifest["sample_ids"] or selected_ids != teacher_ids:
        raise ValueError("validation sample IDs do not match the frozen Teacher order")
    if len(set(selected_ids)) != len(selected_ids) or len(selected_ids) != manifest["record_count"]:
        raise ValueError("validation sample IDs must be unique and match the frozen count")

    source_by_id = {row["id"]: row for row in source_rows}
    if len(source_by_id) != len(source_rows):
        raise ValueError("source validation corpus contains duplicate sample IDs")
    examples: list[DecisionExample] = []
    modality_counts: dict[str, int] = defaultdict(int)
    source_counts: dict[str, int] = defaultdict(int)
    for selected, teacher in zip(selected_rows, teacher_rows, strict=True):
        example = DecisionExample.model_validate(selected)
        source = source_by_id.get(example.id)
        if source is None or any(
            source[field] != selected[field]
            for field in ("options", "target", "source", "modality")
        ):
            raise ValueError(f"snapshot example does not match source validation: {example.id}")
        option_hash = hashlib.sha256(
            json.dumps(example.options, ensure_ascii=False, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()
        if manifest["option_order_sha256"].get(example.id) != option_hash:
            raise ValueError(f"option order hash mismatch for {example.id}")
        if (
            teacher["target"] != example.options.index(example.target)
            or teacher["modality"] != example.modality
            or teacher["source"] != example.source
            or len(teacher["option_probabilities"]) != len(example.options)
        ):
            raise ValueError(f"Teacher target or option metadata mismatch for {example.id}")
        teacher_probabilities = [float(value) for value in teacher["option_probabilities"]]
        if (
            any(not math.isfinite(value) or value < 0 for value in teacher_probabilities)
            or not math.isclose(
                sum(teacher_probabilities), 1.0, rel_tol=1e-5, abs_tol=1e-6
            )
        ):
            raise ValueError(f"Teacher option probabilities are invalid for {example.id}")
        modality_counts[example.modality] += 1
        source_counts[example.source] += 1
        examples.append(example)

    if dict(sorted(modality_counts.items())) != manifest["modality_counts"]:
        raise ValueError("validation modality counts do not match selection manifest")
    if dict(sorted(source_counts.items())) != manifest["source_counts"]:
        raise ValueError("validation source counts do not match selection manifest")
    return examples, teacher_rows


def validate_local_media_paths(
    examples: Iterable[DecisionExample], data_root: str | Path
) -> dict[str, int]:
    """Fail before model loading if any selected example lacks safe local media."""
    root = Path(data_root).resolve()
    counts: dict[str, int] = defaultdict(int)
    for example in examples:
        if example.modality == "text":
            if example.media:
                raise ValueError(f"{example.id}: text example unexpectedly references media")
            continue
        matching = [item for item in example.media if item.kind == example.modality]
        if len(matching) != 1 or len(example.media) != 1:
            raise ValueError(f"{example.id}: expected exactly one matching local media file")
        reference = matching[0]
        if not reference.path:
            raise ValueError(f"{example.id}: media is not locally materialized")
        path = (root / reference.path).resolve()
        if root not in path.parents:
            raise ValueError(f"{example.id}: media path escapes the configured data root")
        if not path.is_file():
            raise FileNotFoundError(f"{example.id}: media file is missing: {path}")
        counts[example.modality] += 1
    return dict(sorted(counts.items()))


def compare_student_predictions(
    student_predictions: list[dict[str, Any]],
    teacher_predictions: list[dict[str, Any]],
    *,
    ece_bins: int = 15,
) -> dict[str, Any]:
    """Compare paired predictions only when sample, target, and option metadata align."""
    if not student_predictions or len(student_predictions) != len(teacher_predictions):
        raise ValueError("student and Teacher predictions must be non-empty and equally sized")
    groups: dict[str, dict[str, list[tuple[list[float], int]]]] = defaultdict(
        lambda: {"student": [], "teacher": []}
    )
    for student, teacher in zip(student_predictions, teacher_predictions, strict=True):
        if (
            student["sample_id"] != teacher["sample_id"]
            or student["target"] != teacher["target"]
            or student["modality"] != teacher["modality"]
            or student["source"] != teacher["source"]
            or len(student["options"]) != len(teacher["option_probabilities"])
        ):
            raise ValueError("student and Teacher predictions are not paired on identical choices")
        target = int(student["target"])
        for name, prediction in (("student", student), ("teacher", teacher)):
            probabilities = [float(value) for value in prediction["option_probabilities"]]
            if (
                any(not math.isfinite(value) or value < 0 for value in probabilities)
                or not math.isclose(sum(probabilities), 1.0, rel_tol=1e-5, abs_tol=1e-6)
            ):
                raise ValueError(f"invalid {name} probabilities for {student['sample_id']}")
            groups["all"][name].append((probabilities, target))
            groups[f"modality:{student['modality']}"][name].append((probabilities, target))
            groups[f"source:{student['source']}"][name].append((probabilities, target))

    output: dict[str, Any] = {}
    for key, paired in sorted(groups.items()):
        student_metrics = _measure(paired["student"], ece_bins=ece_bins)
        teacher_metrics = _measure(paired["teacher"], ece_bins=ece_bins)
        output[key] = {
            "student": student_metrics,
            "teacher": teacher_metrics,
            "delta_student_minus_teacher": {
                metric: float(student_metrics[metric]) - float(teacher_metrics[metric])
                for metric in ("accuracy", "nll", "brier", "ece", "mean_confidence")
            },
        }
    return output


def evaluate_student_examples(
    model: Any,
    processor: Any,
    examples: Iterable[DecisionExample],
    *,
    data_root: Any,
    temperature: float,
    ece_bins: int = 15,
    existing_predictions: list[dict[str, Any]] | None = None,
    on_prediction: Callable[[dict[str, Any]], None] | None = None,
    cache_video_frames: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate examples in input order and report modality/source calibration metrics."""

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
    predictions = list(existing_predictions or [])
    if len(predictions) > len(materialized):
        raise ValueError("existing predictions exceed the validation example count")
    for example, prediction in zip(materialized, predictions, strict=False):
        expected_option_hash = hashlib.sha256(
            json.dumps(example.options, ensure_ascii=False, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()
        if (
            prediction.get("sample_id") != example.id
            or prediction.get("source") != example.source
            or prediction.get("modality") != example.modality
            or prediction.get("target") != example.options.index(example.target)
            or prediction.get("options") != example.options
            or prediction.get("option_order_sha256") != expected_option_hash
        ):
            raise ValueError("existing predictions are not an exact validation prefix")
        probabilities = [float(value) for value in prediction["option_probabilities"]]
        if (
            len(probabilities) != len(example.options)
            or any(not math.isfinite(value) or value < 0 for value in probabilities)
            or not math.isclose(sum(probabilities), 1.0, rel_tol=1e-5, abs_tol=1e-6)
        ):
            raise ValueError(f"existing prediction probabilities are invalid for {example.id}")
        target = int(prediction["target"])
        groups["all"].append((probabilities, target))
        groups[f"modality:{example.modality}"].append((probabilities, target))
        groups[f"source:{example.source}"].append((probabilities, target))
    option_embedding_cache: dict[str, Any] = {}
    video_frame_cache: dict[str, tuple[Any, Any]] | None = (
        {} if cache_video_frames else None
    )
    with torch.inference_mode():
        for example in materialized[len(predictions) :]:
            query_inputs = processor_inputs_for_decision_example(
                processor,
                example,
                data_root=data_root,
                video_frame_cache=video_frame_cache,
            )
            query_embedding = model_sentence_embeddings(model, query_inputs)[0]
            missing_options = list(
                dict.fromkeys(
                    option
                    for option in example.options
                    if decision_option_text(option) not in option_embedding_cache
                )
            )
            if missing_options:
                option_inputs = processor(
                    text=[decision_option_text(option) for option in missing_options],
                    return_tensors="pt",
                )
                new_embeddings = model_sentence_embeddings(model, option_inputs)
                if new_embeddings.shape[0] != len(missing_options):
                    raise ValueError("option encoder returned a different number of embeddings")
                option_embedding_cache.update(
                    {
                        decision_option_text(option): embedding.detach().cpu().float()
                        for option, embedding in zip(
                            missing_options, new_embeddings, strict=True
                        )
                    }
                )
            option_embeddings = torch.stack(
                [
                    option_embedding_cache[decision_option_text(option)]
                    for option in example.options
                ]
            ).to(device=query_embedding.device)
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
            if on_prediction is not None:
                on_prediction(record)

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
