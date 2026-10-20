"""Correctly label the saved paired bootstrap for the PCH LSTM comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def paired_subject_bootstrap(
    probabilities_a: np.ndarray,
    probabilities_b: np.ndarray,
    targets: np.ndarray,
    subject_ids: list[str],
    *,
    name_a: str,
    name_b: str,
    seed: int = 17,
    replicates: int = 5000,
) -> dict[str, Any]:
    if (
        probabilities_a.shape != probabilities_b.shape
        or probabilities_a.ndim != 2
        or probabilities_a.shape[0] != len(targets)
        or len(subject_ids) != len(targets)
        or replicates < 100
    ):
        raise ValueError("bootstrap inputs must be aligned finite probability matrices")
    if (
        not np.isfinite(probabilities_a).all()
        or not np.isfinite(probabilities_b).all()
        or (probabilities_a < 0).any()
        or (probabilities_b < 0).any()
    ):
        raise ValueError("bootstrap probabilities must be finite and nonnegative")
    if not np.allclose(probabilities_a.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("first probability matrix must sum to one")
    if not np.allclose(probabilities_b.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("second probability matrix must sum to one")
    if targets.min(initial=0) < 0 or targets.max(initial=0) >= probabilities_a.shape[1]:
        raise ValueError("target class index is outside the probability matrix")

    subjects = np.asarray(sorted(set(subject_ids)))
    if len(subjects) < 2:
        raise ValueError("participant-cluster bootstrap needs multiple subjects")
    subject_array = np.asarray(subject_ids)
    rows_by_subject = {subject: np.flatnonzero(subject_array == subject) for subject in subjects}
    rng = np.random.default_rng(seed)
    metric_names = ("accuracy", "nll", "brier")
    first_draws = {name: [] for name in metric_names}
    second_draws = {name: [] for name in metric_names}
    delta_draws = {name: [] for name in metric_names}

    def metrics(probabilities: np.ndarray, indices: np.ndarray) -> dict[str, float]:
        selected = probabilities[indices]
        truth = targets[indices]
        row_ids = np.arange(len(indices))
        one_hot = np.zeros_like(selected)
        one_hot[row_ids, truth] = 1.0
        return {
            "accuracy": float((selected.argmax(axis=1) == truth).mean()),
            "nll": float(-np.log(np.clip(selected[row_ids, truth], 1e-12, 1.0)).mean()),
            "brier": float(((selected - one_hot) ** 2).sum(axis=1).mean()),
        }

    for _ in range(replicates):
        sampled_subjects = rng.choice(subjects, size=len(subjects), replace=True)
        indices = np.concatenate([rows_by_subject[subject] for subject in sampled_subjects])
        first = metrics(probabilities_a, indices)
        second = metrics(probabilities_b, indices)
        for name in metric_names:
            first_draws[name].append(first[name])
            second_draws[name].append(second[name])
            delta_draws[name].append(first[name] - second[name])

    def interval(values: list[float]) -> list[float]:
        return [float(value) for value in np.quantile(values, [0.025, 0.975])]

    return {
        "method": "paired participant-cluster bootstrap, percentile 95% interval",
        "seed": seed,
        "replicates": replicates,
        "participants": int(len(subjects)),
        "first_model": name_a,
        "second_model": name_b,
        "delta_direction": f"{name_a} minus {name_b}",
        "first_model_intervals": {name: interval(values) for name, values in first_draws.items()},
        "second_model_intervals": {name: interval(values) for name, values in second_draws.items()},
        "delta_intervals": {name: interval(values) for name, values in delta_draws.items()},
    }


def analyze_run(run_dir: Path, output_path: Path) -> dict[str, Any]:
    result_path = run_dir / "result.json"
    metadata_path = run_dir / "run-metadata.json"
    predictions_path = run_dir / "validation-predictions.jsonl"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("result_sha256") != sha256_file(result_path):
        raise ValueError("run result does not match its completion metadata hash")
    if result.get("status") != "PASS_PCH_LSTM_SENSOR_DEV_COMPARISON":
        raise ValueError("completed PCH LSTM development run is required")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite correction: {output_path}")
    rows = [json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines()]
    validation = result["validation"]
    if len(rows) != validation["records"] or len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("saved validation predictions are incomplete or duplicated")
    if len({row["subject_id"] for row in rows}) != validation["unique_participants"]:
        raise ValueError("saved validation participant count differs from the run receipt")
    classes = result["split"]["classes"]
    class_to_id = {label: index for index, label in enumerate(classes)}
    try:
        targets = np.asarray([class_to_id[row["target_id"]] for row in rows], dtype=np.int64)
    except KeyError as exc:
        raise ValueError("saved prediction has an unknown target class") from exc
    lstm_probs = np.asarray([row["lstm_probabilities"] for row in rows], dtype=np.float64)
    moment_probs = np.asarray([row["moment_probabilities"] for row in rows], dtype=np.float64)
    bootstrap = paired_subject_bootstrap(
        lstm_probs,
        moment_probs,
        targets,
        [row["subject_id"] for row in rows],
        name_a="window-level LSTM",
        name_b="frozen MOMENT + linear readout",
    )
    correction = {
        "status": "PASS_POSTHOC_BOOTSTRAP_LABEL_CORRECTION",
        "purpose": (
            "Recompute paired validation intervals from saved predictions; no inference or training"
        ),
        "run_result_sha256": sha256_file(result_path),
        "validation_predictions_sha256": sha256_file(predictions_path),
        "validation_id_order_sha256": result["validation_id_order_sha256"],
        "examples": len(rows),
        "heldout_payload_opened": False,
        "sealed_audit_loaded": False,
        "original_bootstrap_direction_text": validation[
            "participant_cluster_bootstrap_lstm_minus_moment"
        ]["delta_direction"],
        "corrected_bootstrap": bootstrap,
        "note": (
            "The original numeric bootstrap was generated by a MOMENT-vs-static helper; its "
            "numeric values are not treated as authoritative. These intervals are recomputed "
            "from saved LSTM and MOMENT probabilities with explicit model names."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(correction, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return correction


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze_run(args.run_dir, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
