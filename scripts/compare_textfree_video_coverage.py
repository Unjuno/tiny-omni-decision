"""Compare nested CLEVRER coverage fits on one frozen validation scene set."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

ARMS = ("mean_mlp", "bilstm")
METRICS = ("accuracy", "nll", "brier", "ece_15")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError(f"empty JSONL: {path}")
    return rows


def _probability_arrays(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    correct, nll, brier, confidence, types, scenes = [], [], [], [], [], []
    ids: set[str] = set()
    for row in rows:
        required = {"id", "target", "prediction", "options", "question_type", "scene_group_id"}
        if not required.issubset(row) or row["id"] in ids:
            raise ValueError("prediction row is missing required fields or repeats an ID")
        ids.add(row["id"])
        options = row["options"]
        option_ids = [option["option_id"] for option in options]
        probabilities = np.asarray([option["probability"] for option in options], dtype=np.float64)
        if (
            len(option_ids) < 2
            or len(set(option_ids)) != len(option_ids)
            or row["target"] not in option_ids
            or row["prediction"] not in option_ids
            or not np.isfinite(probabilities).all()
            or (probabilities < 0).any()
            or not math.isclose(float(probabilities.sum()), 1.0, abs_tol=1e-5, rel_tol=0.0)
        ):
            raise ValueError("prediction probabilities or target are invalid")
        target_index = option_ids.index(row["target"])
        predicted_index = int(np.argmax(probabilities))
        if option_ids[predicted_index] != row["prediction"]:
            raise ValueError("saved prediction does not match the supplied-option argmax")
        correct.append(row["prediction"] == row["target"])
        nll.append(-math.log(max(float(probabilities[target_index]), 1e-15)))
        brier.append(float(((probabilities - np.eye(len(option_ids))[target_index]) ** 2).sum()))
        confidence.append(float(probabilities[predicted_index]))
        types.append(str(row["question_type"]))
        scenes.append(str(row["scene_group_id"]))
    return {
        "correct": np.asarray(correct, dtype=np.float64),
        "nll": np.asarray(nll, dtype=np.float64),
        "brier": np.asarray(brier, dtype=np.float64),
        "confidence": np.asarray(confidence, dtype=np.float64),
        "types": np.asarray(types),
        "scenes": np.asarray(scenes),
        "ids": np.asarray([row["id"] for row in rows]),
    }


def _metric_values(arrays: dict[str, np.ndarray], indexes: np.ndarray) -> dict[str, float]:
    if indexes.size == 0:
        raise ValueError("cannot calculate metrics for an empty subset")
    correct = arrays["correct"][indexes]
    confidence = arrays["confidence"][indexes]
    bins = np.minimum((confidence * 15).astype(np.int64), 14)
    ece = 0.0
    for bin_id in range(15):
        in_bin = bins == bin_id
        if in_bin.any():
            bin_accuracy = float(correct[in_bin].mean())
            bin_confidence = float(confidence[in_bin].mean())
            ece += float(in_bin.mean()) * abs(bin_accuracy - bin_confidence)
    return {
        "accuracy": float(correct.mean()),
        "nll": float(arrays["nll"][indexes].mean()),
        "brier": float(arrays["brier"][indexes].mean()),
        "ece_15": ece,
    }


def compare_prediction_sets(
    small: dict[str, list[dict[str, Any]]],
    large: dict[str, list[dict[str, Any]]],
    *,
    bootstrap_repetitions: int = 5000,
    seed: int = 17,
) -> dict[str, Any]:
    if set(small) != set(ARMS) or set(large) != set(ARMS):
        raise ValueError("both coverage sizes require mean_mlp and bilstm predictions")
    arrays: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for size, predictions in (("small", small), ("large", large)):
        for arm, rows in predictions.items():
            arrays[(size, arm)] = _probability_arrays(rows)
    base_ids = arrays[("small", ARMS[0])]["ids"]
    base_scenes = arrays[("small", ARMS[0])]["scenes"]
    base_types = arrays[("small", ARMS[0])]["types"]
    for size in ("small", "large"):
        for arm in ARMS:
            current = arrays[(size, arm)]
            if not np.array_equal(current["ids"], base_ids):
                raise ValueError("all fits must preserve the same validation ID order")
            if not np.array_equal(current["scenes"], base_scenes) or not np.array_equal(
                current["types"], base_types
            ):
                raise ValueError("paired validation scene/task metadata differs")
    if bootstrap_repetitions < 1:
        raise ValueError("bootstrap_repetitions must be positive")

    question_types = sorted(set(base_types.tolist()))
    observed: dict[str, Any] = {}
    for arm in ARMS:
        by_type: dict[str, Any] = {}
        for question_type in question_types:
            indexes = np.flatnonzero(base_types == question_type)
            by_type[question_type] = {
                size: _metric_values(arrays[(size, arm)], indexes)
                for size in ("small", "large")
            }
        macro = {
            size: {
                metric: float(
                    np.mean([by_type[task][size][metric] for task in question_types])
                )
                for metric in METRICS
            }
            for size in ("small", "large")
        }
        observed[arm] = {"by_question_type": by_type, "macro": macro}

    scene_to_indexes = {
        scene: np.flatnonzero(base_scenes == scene) for scene in sorted(set(base_scenes.tolist()))
    }
    rng = np.random.default_rng(seed)
    bootstrap: dict[str, Any] = {}
    scene_names = np.asarray(list(scene_to_indexes))
    for arm in ARMS:
        records: dict[str, dict[str, list[float]]] = {
            task: {metric: [] for metric in METRICS} for task in question_types
        }
        records["macro"] = {metric: [] for metric in METRICS}
        for _ in range(bootstrap_repetitions):
            drawn = rng.choice(scene_names, size=len(scene_names), replace=True)
            indexes = np.concatenate([scene_to_indexes[str(scene)] for scene in drawn])
            for task in question_types:
                task_indexes = indexes[base_types[indexes] == task]
                small_values = _metric_values(arrays[("small", arm)], task_indexes)
                large_values = _metric_values(arrays[("large", arm)], task_indexes)
                for metric in METRICS:
                    records[task][metric].append(large_values[metric] - small_values[metric])
            for metric in METRICS:
                records["macro"][metric].append(
                    float(np.mean([records[task][metric][-1] for task in question_types]))
                )
        bootstrap[arm] = {
            group: {
                metric: {
                    "large_minus_small": float(np.mean(values)),
                    "percentile_95_ci": [
                        float(np.quantile(values, 0.025, method="linear")),
                        float(np.quantile(values, 0.975, method="linear")),
                    ],
                }
                for metric, values in metrics.items()
            }
            for group, metrics in records.items()
        }
    return {
        "status": "PASS_PAIRED_NESTED_COVERAGE_COMPARISON",
        "paired_validation_examples": len(base_ids),
        "paired_validation_scenes": len(scene_names),
        "question_types": question_types,
        "observed_metrics": observed,
        "scene_bootstrap": {
            "cluster": "scene_group_id",
            "repetitions": bootstrap_repetitions,
            "seed": seed,
            "interval": "percentile 95%; NumPy linear quantile",
            "note": "development validation also selected checkpoints; intervals are descriptive",
            "large_minus_small": bootstrap,
        },
    }


def load_run(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    report_path = run_dir / "run-report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("status") != "PASS_TEXTFREE_VIDEO_PAIRED_FIT"
        or report.get("sealed_audit_loaded") is not False
        or report.get("validation_used_for_checkpoint_selection") is not True
    ):
        raise ValueError(f"unexpected or non-development run report: {run_dir}")
    preflight_path = run_dir / "input-preflight.json"
    if _sha256(preflight_path) != report.get("input_preflight_sha256"):
        raise ValueError(f"input preflight hash mismatch: {run_dir}")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    sample_dir = Path(preflight["sample_dir"])
    validation_path = sample_dir / "validation.jsonl"
    if _sha256(validation_path) != preflight.get("validation_jsonl_sha256"):
        raise ValueError("frozen validation manifest hash mismatch")
    manifest = _read_jsonl(validation_path)
    predictions: dict[str, list[dict[str, Any]]] = {}
    for arm in ARMS:
        path = run_dir / f"selected-validation-{arm}.jsonl"
        if _sha256(path) != report["runs"][arm]["selected_prediction_sha256"]:
            raise ValueError(f"selected validation prediction hash mismatch: {path}")
        predictions[arm] = _read_jsonl(path)
        if [row["id"] for row in predictions[arm]] != [row["id"] for row in manifest]:
            raise ValueError("saved predictions do not match frozen validation ID order")
    return {"run_report": report, "preflight": preflight}, predictions


def build_report(small_dir: Path, large_dir: Path, repetitions: int) -> dict[str, Any]:
    small_metadata, small_predictions = load_run(small_dir)
    large_metadata, large_predictions = load_run(large_dir)
    small_preflight = small_metadata["preflight"]
    large_preflight = large_metadata["preflight"]
    if small_preflight["validation_jsonl_sha256"] != large_preflight["validation_jsonl_sha256"]:
        raise ValueError("coverage fits must use the exact same frozen validation manifest")
    small_train_path = Path(small_preflight["sample_dir"]) / "train.jsonl"
    large_train_path = Path(large_preflight["sample_dir"]) / "train.jsonl"
    small_train, large_train = _read_jsonl(small_train_path), _read_jsonl(large_train_path)
    small_scene_ids = {int(row["scene_index"]) for row in small_train}
    large_scene_ids = {int(row["scene_index"]) for row in large_train}
    small_ids = {row["id"] for row in small_train}
    large_ids = {row["id"] for row in large_train}
    if not small_scene_ids < large_scene_ids or not small_ids < large_ids:
        raise ValueError("small training split must be a strict scene-nested subset of large")
    if not small_ids.issubset(large_ids):
        raise ValueError("small training sample IDs are not nested in large training IDs")
    comparison = compare_prediction_sets(
        small_predictions,
        large_predictions,
        bootstrap_repetitions=repetitions,
        seed=17,
    )
    comparison["training_coverage"] = {
        "small": {
            "examples": len(small_train),
            "unique_scenes": len(small_scene_ids),
            "unique_media": len({row["media_sha256"] for row in small_train}),
            "manifest_sha256": _sha256(small_train_path),
            "run_report_sha256": _sha256(small_dir / "run-report.json"),
            "effective_config_sha256": small_metadata["run_report"][
                "effective_config_sha256"
            ],
        },
        "large": {
            "examples": len(large_train),
            "unique_scenes": len(large_scene_ids),
            "unique_media": len({row["media_sha256"] for row in large_train}),
            "manifest_sha256": _sha256(large_train_path),
            "run_report_sha256": _sha256(large_dir / "run-report.json"),
            "effective_config_sha256": large_metadata["run_report"][
                "effective_config_sha256"
            ],
        },
        "nested_scene_subset": True,
        "same_validation_manifest_sha256": small_preflight["validation_jsonl_sha256"],
    }
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--small-run-dir", type=Path, required=True)
    parser.add_argument("--large-run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    args = parser.parse_args()
    result = build_report(args.small_run_dir, args.large_run_dir, args.bootstrap_repetitions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite coverage comparison: {args.output}")
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"PASS_PAIRED_NESTED_COVERAGE_COMPARISON {args.output}")


if __name__ == "__main__":
    main()
