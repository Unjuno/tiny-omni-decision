"""Post-hoc error analysis for saved UCI HAR validation predictions.

This module only consumes already saved validation predictions and provenance
files. It never opens the official test split or loads a model/checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

ACTIVITIES = (
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("prediction rows must not be empty")
    n = len(rows)
    correct = 0
    nll = 0.0
    brier = 0.0
    confidence = []
    class_counts: dict[str, dict[str, Any]] = {
        activity: {"support": 0, "correct": 0, "recall": None} for activity in ACTIVITIES
    }
    confusion = {truth: {pred: 0 for pred in ACTIVITIES} for truth in ACTIVITIES}
    true_probability = 0.0
    errors = 0
    error_confidence = 0.0
    bins: list[list[tuple[bool, float]]] = [[] for _ in range(10)]

    for row in rows:
        truth = str(row["target"])
        prediction = str(row["prediction"])
        probs = row["probabilities"]
        if truth not in class_counts or prediction not in class_counts:
            raise ValueError("prediction row contains an unknown activity")
        values = [float(probs[activity]) for activity in ACTIVITIES]
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("probabilities must be finite and nonnegative")
        if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-5):
            raise ValueError("probabilities must sum to one within 1e-5")
        expected_prediction = ACTIVITIES[max(range(len(values)), key=values.__getitem__)]
        if prediction != expected_prediction:
            raise ValueError("saved prediction disagrees with probability argmax")

        is_correct = prediction == truth
        conf = max(values)
        true_prob = values[ACTIVITIES.index(truth)]
        correct += int(is_correct)
        nll -= math.log(max(true_prob, 1e-12))
        brier += sum(
            (value - float(activity == truth)) ** 2
            for activity, value in zip(ACTIVITIES, values, strict=True)
        )
        true_probability += true_prob
        confidence.append(conf)
        errors += int(not is_correct)
        error_confidence += conf if not is_correct else 0.0
        class_counts[truth]["support"] += 1
        class_counts[truth]["correct"] += int(is_correct)
        confusion[truth][prediction] += 1
        bin_index = min(int(conf * 10), 9)
        bins[bin_index].append((is_correct, conf))

    recalls = []
    for activity in ACTIVITIES:
        entry = class_counts[activity]
        if entry["support"]:
            entry["recall"] = entry["correct"] / entry["support"]
            recalls.append(entry["recall"])
        del entry["correct"]

    ece = sum(
        len(items)
        / n
        * abs(
            sum(correctness for correctness, _ in items) / len(items)
            - sum(conf for _, conf in items) / len(items)
        )
        for items in bins
        if items
    )
    return {
        "count": n,
        "accuracy": correct / n,
        "macro_accuracy": sum(recalls) / len(recalls),
        "nll": nll / n,
        "brier": brier / n,
        "ece_10_equal_width": ece,
        "mean_confidence": sum(confidence) / n,
        "mean_true_class_probability": true_probability / n,
        "error_count": errors,
        "mean_confidence_on_errors": error_confidence / errors if errors else None,
        "per_activity": class_counts,
        "confusion": confusion,
    }


def analyze_paired_predictions(
    static_rows: list[dict[str, Any]], lstm_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Compare paired held-out validation predictions by class and subject."""
    if len(static_rows) != len(lstm_rows) or not static_rows:
        raise ValueError("static and LSTM predictions must have the same nonzero row count")
    static_by_id = {str(row["sample_id"]): row for row in static_rows}
    lstm_by_id = {str(row["sample_id"]): row for row in lstm_rows}
    if len(static_by_id) != len(static_rows) or len(lstm_by_id) != len(lstm_rows):
        raise ValueError("sample IDs must be unique within each prediction file")
    if list(static_by_id) != list(lstm_by_id):
        raise ValueError("static and LSTM sample IDs/order differ")

    for sample_id in static_by_id:
        static = static_by_id[sample_id]
        lstm = lstm_by_id[sample_id]
        if static["target"] != lstm["target"] or int(static["subject_id"]) != int(
            lstm["subject_id"]
        ):
            raise ValueError(f"paired prediction metadata differs for {sample_id}")

    subject_rows: dict[int, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    class_transitions: dict[str, Counter[str]] = {activity: Counter() for activity in ACTIVITIES}
    pair_outcomes: Counter[str] = Counter()
    for sample_id in static_by_id:
        static = static_by_id[sample_id]
        lstm = lstm_by_id[sample_id]
        subject_rows.setdefault(int(static["subject_id"]), []).append((static, lstm))
        truth = str(static["target"])
        static_ok = static["prediction"] == truth
        lstm_ok = lstm["prediction"] == truth
        pair_outcomes[
            "both_correct"
            if static_ok and lstm_ok
            else "static_only_correct"
            if static_ok
            else "lstm_only_correct"
            if lstm_ok
            else "both_wrong"
        ] += 1
        class_transitions[truth][f"{static['prediction']}->{lstm['prediction']}"] += 1

    by_subject = {}
    for subject, pairs in sorted(subject_rows.items()):
        static_subject = [pair[0] for pair in pairs]
        lstm_subject = [pair[1] for pair in pairs]
        static_metrics = _metrics(static_subject)
        lstm_metrics = _metrics(lstm_subject)
        by_subject[str(subject)] = {
            "count": len(pairs),
            "static_accuracy": static_metrics["accuracy"],
            "lstm_accuracy": lstm_metrics["accuracy"],
            "lstm_minus_static_accuracy": lstm_metrics["accuracy"] - static_metrics["accuracy"],
            "static_macro_accuracy": static_metrics["macro_accuracy"],
            "lstm_macro_accuracy": lstm_metrics["macro_accuracy"],
            "class_support": {
                name: static_metrics["per_activity"][name]["support"] for name in ACTIVITIES
            },
            "static_per_activity_recall": {
                name: static_metrics["per_activity"][name]["recall"] for name in ACTIVITIES
            },
            "lstm_per_activity_recall": {
                name: lstm_metrics["per_activity"][name]["recall"] for name in ACTIVITIES
            },
        }

    return {
        "count": len(static_rows),
        "static": _metrics(static_rows),
        "lstm": _metrics(lstm_rows),
        "paired_outcomes": dict(pair_outcomes),
        "paired_accuracy_delta_lstm_minus_static": _metrics(lstm_rows)["accuracy"]
        - _metrics(static_rows)["accuracy"],
        "by_true_activity_prediction_transition": {
            activity: dict(sorted(counts.items())) for activity, counts in class_transitions.items()
        },
        "by_subject": by_subject,
        "confidence_intervals": None,
        "confidence_interval_note": (
            "Only four held-out participants; report individual participant results "
            "without a resampling interval."
        ),
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def analyze_run(run_dir: Path, output_path: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    output_path = output_path.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite analysis result: {output_path}")
    required = (
        "candidate-freeze.json",
        "artifact-index.json",
        "split-manifest.json",
        "static-best-validation-predictions.jsonl",
        "static-best-validation-reload.jsonl",
        "lstm-best-validation-predictions.jsonl",
        "lstm-best-validation-reload.jsonl",
    )
    for filename in required:
        if not (run_dir / filename).is_file():
            raise FileNotFoundError(run_dir / filename)

    freeze = _read_json(run_dir / "candidate-freeze.json")
    index = _read_json(run_dir / "artifact-index.json")
    split = _read_json(run_dir / "split-manifest.json")
    if freeze.get("frozen_before_test_read") is not True:
        raise ValueError("candidate freeze does not prove it preceded test access")
    if split.get("official_test_split_loaded") is not False:
        raise ValueError("refusing analysis: test-split status is not explicitly closed")
    if split.get("split_manifest_sha256") != freeze.get("split_manifest_sha256"):
        raise ValueError("split-manifest identity differs from candidate freeze")
    artifacts = {item["path"]: item for item in index["artifacts"]}
    for name in required:
        if name in artifacts:
            record = artifacts[name]
            path = run_dir / name
            if path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
                raise ValueError(f"artifact-index integrity check failed for {name}")
    for kind in ("static", "lstm"):
        pred_name = f"{kind}-best-validation-predictions.jsonl"
        reload_name = f"{kind}-best-validation-reload.jsonl"
        if sha256_file(run_dir / pred_name) != sha256_file(run_dir / reload_name):
            raise ValueError(f"best-checkpoint reload predictions differ for {kind}")
        candidate = freeze.get("candidates", {}).get(kind, {})
        if candidate.get("selector") != "earliest minimum validation NLL":
            raise ValueError(f"unexpected validation checkpoint selector for {kind}")

    static_rows = _read_jsonl(run_dir / "static-best-validation-predictions.jsonl")
    lstm_rows = _read_jsonl(run_dir / "lstm-best-validation-predictions.jsonl")
    validation_subjects = sorted(int(x) for x in split["validation_subjects"])
    observed_subjects = sorted({int(row["subject_id"]) for row in static_rows})
    if observed_subjects != validation_subjects:
        raise ValueError("validation predictions do not match frozen validation participants")
    train_subjects = {int(x) for x in split["train_subjects"]}
    if train_subjects & set(validation_subjects):
        raise ValueError("frozen train and validation participant groups overlap")
    expected_ids = split.get("validation_sample_ids")
    observed_ids = [str(row["sample_id"]) for row in static_rows]
    if expected_ids is None or observed_ids != expected_ids:
        raise ValueError("validation predictions differ from frozen validation sample IDs/order")
    result = {
        "analysis_id": "uci-har-sensor-v1-attempt-02-long100-validation-error-analysis",
        "status": "completed_posthoc_development_validation_analysis",
        "run_directory": str(run_dir),
        "source_commit": freeze.get("source_commit"),
        "source_archive_sha256": freeze.get("source_archive_sha256"),
        "split_manifest_sha256": freeze.get("split_manifest_sha256"),
        "candidate_freeze_sha256": sha256_file(run_dir / "candidate-freeze.json"),
        "artifact_index_sha256": sha256_file(run_dir / "artifact-index.json"),
        "validation_prediction_sha256": {
            "static": sha256_file(run_dir / "static-best-validation-predictions.jsonl"),
            "lstm": sha256_file(run_dir / "lstm-best-validation-predictions.jsonl"),
        },
        "checkpoint_sha256": {
            kind: freeze["candidates"][kind]["checkpoint_sha256"] for kind in ("static", "lstm")
        },
        "best_epoch": {
            kind: freeze["candidates"][kind]["best_epoch"] for kind in ("static", "lstm")
        },
        "validation_subjects": validation_subjects,
        "official_test_opened": False,
        "model_loaded": False,
        "training_performed": False,
        "sealed_audit_opened": False,
        "interpretation_limit": (
            "Post-hoc diagnostic on validation used for checkpoint selection; not an "
            "independent quality estimate or a checkpoint-selection input."
        ),
        "comparison": analyze_paired_predictions(static_rows, lstm_rows),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return result
