"""One-time descriptive error breakdown of frozen UCI HAR validation predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from scripts.uci_har_sensor_experiment import ACTIVITIES, _metrics

PREDICTION_FILES = {
    "static": "static-best-validation-predictions.jsonl",
    "lstm": "lstm-best-validation-predictions.jsonl",
}
EXPECTED_EPOCHS = {"static": 84, "lstm": 15}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_prediction_file(path: Path) -> list[dict[str, Any]]:
    rows = []
    identifiers = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        row = json.loads(line)
        sample_id = str(row["sample_id"])
        if sample_id in identifiers:
            raise ValueError(f"duplicate sample ID in {path.name}: {sample_id}")
        identifiers.add(sample_id)
        target = str(row["target"])
        predicted = str(row["prediction"])
        if target not in ACTIVITIES or predicted not in ACTIVITIES:
            raise ValueError(f"unknown activity label at {path.name}:{line_number}")
        probabilities = row["probabilities"]
        if set(probabilities) != set(ACTIVITIES):
            raise ValueError(f"activity probability keys differ at {path.name}:{line_number}")
        values = np.asarray([probabilities[name] for name in ACTIVITIES], dtype=np.float64)
        if (not np.isfinite(values).all() or (values < 0).any()
                or not np.isclose(values.sum(), 1.0, atol=1e-6)):
            raise ValueError(f"invalid probabilities at {path.name}:{line_number}")
        if ACTIVITIES[int(values.argmax())] != predicted:
            raise ValueError(
                "prediction does not match probability argmax "
                f"at {path.name}:{line_number}"
            )
        subject_id = int(row["subject_id"])
        if subject_id < 1:
            raise ValueError(f"invalid subject ID at {path.name}:{line_number}")
        rows.append({
            "sample_id": sample_id,
            "subject_id": subject_id,
            "target": target,
            "prediction": predicted,
            "probabilities": values,
        })
    if not rows:
        raise ValueError(f"no prediction rows in {path}")
    return rows


def _arrays(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    targets = np.asarray([ACTIVITIES.index(row["target"]) for row in rows], dtype=np.int64)
    subjects = np.asarray([row["subject_id"] for row in rows], dtype=np.int64)
    probabilities = np.stack([row["probabilities"] for row in rows])
    return targets, subjects, probabilities, probabilities.argmax(axis=1)


def _confusion_metrics(targets: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    predicted = probabilities.argmax(axis=1)
    matrix = np.zeros((len(ACTIVITIES), len(ACTIVITIES)), dtype=np.int64)
    np.add.at(matrix, (targets, predicted), 1)
    details = {}
    for index, activity in enumerate(ACTIVITIES):
        support = int(matrix[index].sum())
        predicted_support = int(matrix[:, index].sum())
        true_positive = int(matrix[index, index])
        details[activity] = {
            "support": support,
            "correct": true_positive,
            "recall": float(true_positive / support) if support else None,
            "predicted_support": predicted_support,
            "precision": (
                float(true_positive / predicted_support) if predicted_support else None
            ),
        }
    base = _metrics(probabilities, targets)
    return {
        **base,
        "confusion_matrix": matrix.tolist(),
        "confusion_rows_true_columns_predicted": list(ACTIVITIES),
        "per_activity": details,
    }


def analyze_predictions(
    static_rows: list[dict[str, Any]], lstm_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    static_ids = [row["sample_id"] for row in static_rows]
    lstm_ids = [row["sample_id"] for row in lstm_rows]
    if static_ids != lstm_ids:
        raise ValueError("static and LSTM sample IDs/order differ")
    for key in ("target", "subject_id"):
        if [row[key] for row in static_rows] != [row[key] for row in lstm_rows]:
            raise ValueError(f"static and LSTM {key} values differ")

    static_targets, static_subjects, static_probabilities, static_predictions = _arrays(
        static_rows
    )
    lstm_targets, lstm_subjects, lstm_probabilities, lstm_predictions = _arrays(lstm_rows)
    if not np.array_equal(static_targets, lstm_targets) or not np.array_equal(
        static_subjects, lstm_subjects
    ):
        raise ValueError("paired target or subject arrays differ")

    by_subject = {}
    for subject_id in sorted(set(int(value) for value in static_subjects)):
        mask = static_subjects == subject_id
        by_subject[str(subject_id)] = {
            "examples": int(mask.sum()),
            "static": _metrics(static_probabilities[mask], static_targets[mask]),
            "lstm": _metrics(lstm_probabilities[mask], lstm_targets[mask]),
            "paired_correctness": {
                "both_correct": int(((static_predictions == static_targets)
                                     & (lstm_predictions == lstm_targets) & mask).sum()),
                "static_only_correct": int(((static_predictions == static_targets)
                                            & (lstm_predictions != lstm_targets) & mask).sum()),
                "lstm_only_correct": int(((static_predictions != static_targets)
                                          & (lstm_predictions == lstm_targets) & mask).sum()),
                "both_wrong": int(((static_predictions != static_targets)
                                   & (lstm_predictions != lstm_targets) & mask).sum()),
            },
        }

    per_subject_activity = defaultdict(lambda: defaultdict(lambda: {
        "examples": 0,
        "static_correct": 0,
        "lstm_correct": 0,
        "static_to_lstm_confusions": defaultdict(int),
    }))
    for index, row in enumerate(static_rows):
        subject = str(row["subject_id"])
        target = str(row["target"])
        record = per_subject_activity[subject][target]
        record["examples"] += 1
        record["static_correct"] += int(static_predictions[index] == static_targets[index])
        record["lstm_correct"] += int(lstm_predictions[index] == lstm_targets[index])
        if lstm_predictions[index] != lstm_targets[index]:
            confusion = ACTIVITIES[int(lstm_predictions[index])]
            record["static_to_lstm_confusions"][confusion] += 1

    serialized_subject_activity = {}
    for subject, activities in sorted(per_subject_activity.items()):
        serialized_subject_activity[subject] = {}
        for activity, values in sorted(activities.items()):
            values["static_to_lstm_confusions"] = dict(
                sorted(values["static_to_lstm_confusions"].items())
            )
            serialized_subject_activity[subject][activity] = dict(values)

    return {
        "examples": len(static_rows),
        "subjects": sorted(set(int(value) for value in static_subjects)),
        "sample_ids_sha256": hashlib.sha256("\n".join(static_ids).encode()).hexdigest(),
        "static": _confusion_metrics(static_targets, static_probabilities),
        "lstm": _confusion_metrics(lstm_targets, lstm_probabilities),
        "per_subject": by_subject,
        "paired_by_subject_and_target": serialized_subject_activity,
    }


def analyze_run(run_dir: Path) -> dict[str, Any]:
    index_path = run_dir / "artifact-index.json"
    freeze_path = run_dir / "candidate-freeze.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("frozen_before_test_read") is not True:
        raise ValueError("candidate freeze does not establish pre-test checkpoint selection")
    if freeze.get("official_test_loaded") is True:
        raise ValueError("refusing to analyze a run marked as having loaded official test")
    indexed = {row["path"]: row for row in index["artifacts"]}
    if index.get("artifact_count") != len(indexed):
        raise ValueError("artifact index count or paths are inconsistent")

    predictions = {}
    verified_hashes = {}
    for model, filename in PREDICTION_FILES.items():
        checkpoint_name = f"{model}-best.safetensors"
        for relative in (filename, checkpoint_name):
            record = indexed.get(relative)
            path = run_dir / relative
            if record is None or not path.is_file():
                raise ValueError(f"missing indexed artifact: {relative}")
            if path.stat().st_size != int(record["bytes"]):
                raise ValueError(f"byte-size mismatch for indexed artifact: {relative}")
            digest = sha256_file(path)
            if digest != record["sha256"]:
                raise ValueError(f"SHA-256 mismatch for indexed artifact: {relative}")
            verified_hashes[relative] = digest
        candidate = freeze["candidates"][model]
        if candidate["best_epoch"] != EXPECTED_EPOCHS[model]:
            raise ValueError(f"selected epoch changed for {model}")
        if candidate["checkpoint_sha256"] != verified_hashes[checkpoint_name]:
            raise ValueError(f"frozen checkpoint hash differs for {model}")
        predictions[model] = read_prediction_file(run_dir / filename)

    report = analyze_predictions(predictions["static"], predictions["lstm"])
    report["provenance"] = {
        "run_id": index["run_id"],
        "artifact_index_sha256": sha256_file(index_path),
        "candidate_freeze_sha256": sha256_file(freeze_path),
        "verified_artifact_hashes": verified_hashes,
        "selected_epochs": EXPECTED_EPOCHS,
        "official_test_loaded": False,
        "source_signals_loaded": False,
        "model_fitted_or_loaded": False,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"refusing to overwrite nonempty output: {args.output_dir}")
    report = analyze_run(args.run_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "validation-error-analysis.json"
    output_path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n",
                           encoding="utf-8", newline="\n")
    print(json.dumps({
        "output": str(output_path),
        "examples": report["examples"],
        "subjects": report["subjects"],
        "static_accuracy": report["static"]["accuracy"],
        "lstm_accuracy": report["lstm"]["accuracy"],
        "official_test_loaded": False,
        "source_signals_loaded": False,
    }, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
