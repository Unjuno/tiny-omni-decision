"""Diagnose logit-scale drift on saved native full-extent validation predictions.

This is a read-only analysis: it does not load a model, fit calibration, train,
select checkpoints, or access sealed audit data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

SCENE_ID = re.compile(r"^video:clevrer-validation-(\d+)-q\d+$")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def per_example(logits: np.ndarray, records: list[dict]) -> dict[str, np.ndarray]:
    if logits.ndim != 2 or logits.shape[0] != len(records):
        raise ValueError("prediction array and validation IDs differ in shape")
    accuracy, confidence, nll, brier = [], [], [], []
    span, margin, absmax = [], [], []
    for row, record in zip(logits, records, strict=True):
        option_count = len(record["option_ids"])
        target = record["target_index"]
        if not 2 <= option_count <= logits.shape[1] or not 0 <= target < option_count:
            raise ValueError("invalid option count or target index")
        values = np.asarray(row[:option_count], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("non-finite saved validation logits")
        shifted = values - values.max()
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum()
        top = int(probabilities.argmax())
        one_hot = np.zeros(option_count, dtype=np.float64)
        one_hot[target] = 1.0
        ordered = np.sort(values)
        accuracy.append(float(top == target))
        confidence.append(float(probabilities[top]))
        nll.append(float(np.log(np.exp(shifted).sum()) - shifted[target]))
        brier.append(float(np.square(probabilities - one_hot).sum()))
        span.append(float(ordered[-1] - ordered[0]))
        margin.append(float(ordered[-1] - ordered[-2]))
        absmax.append(float(np.max(np.abs(values))))
    return {
        "accuracy": np.asarray(accuracy),
        "confidence": np.asarray(confidence),
        "nll": np.asarray(nll),
        "brier": np.asarray(brier),
        "logit_span": np.asarray(span),
        "top_two_margin": np.asarray(margin),
        "abs_logit_max": np.asarray(absmax),
    }


def summarize(rows: dict[str, np.ndarray]) -> dict[str, float | int]:
    accuracy = rows["accuracy"]
    confidence = rows["confidence"]
    ece = 0.0
    for index in range(10):
        low, high = index / 10, (index + 1) / 10
        selected = (confidence >= low) & (confidence < high if index < 9 else confidence <= high)
        if selected.any():
            ece += selected.mean() * abs(accuracy[selected].mean() - confidence[selected].mean())
    return {
        "n": int(accuracy.size),
        "accuracy": float(accuracy.mean()),
        "nll": float(rows["nll"].mean()),
        "brier": float(rows["brier"].mean()),
        "ece_10bin": float(ece),
        "mean_confidence": float(confidence.mean()),
        "median_logit_span": float(np.median(rows["logit_span"])),
        "p90_logit_span": float(np.quantile(rows["logit_span"], 0.90)),
        "median_top_two_margin": float(np.median(rows["top_two_margin"])),
        "p95_abs_logit_max": float(np.quantile(rows["abs_logit_max"], 0.95)),
        "wrong_count": int((accuracy == 0).sum()),
        "mean_confidence_when_wrong": (
            float(confidence[accuracy == 0].mean()) if (accuracy == 0).any() else 0.0
        ),
    }


def paired_scene_bootstrap(
    differences: np.ndarray,
    scene_ids: list[str],
    *,
    iterations: int = 10_000,
    seed: int = 17,
) -> dict[str, object]:
    if differences.shape != (len(scene_ids),) or iterations < 1:
        raise ValueError("scene bootstrap input mismatch")
    names = sorted(set(scene_ids))
    if len(names) < 2:
        raise ValueError("scene bootstrap needs at least two independent groups")
    by_scene = {name: differences[np.asarray(scene_ids) == name] for name in names}
    scene_means = np.asarray([by_scene[name].mean() for name in names])
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(names), size=(iterations, len(names)))
    boot = scene_means[samples].mean(axis=1)
    return {
        "group": "CLEVRER scene ID parsed from pinned validation ID",
        "scene_count": len(names),
        "query_count": len(scene_ids),
        "mean_scene_paired_difference": float(scene_means.mean()),
        "percentile_95_interval": [float(x) for x in np.quantile(boot, [0.025, 0.975])],
        "iterations": iterations,
        "seed": seed,
        "method": "resample scene means with replacement; percentile interval",
    }


def analyze(run_dir: Path, steps: list[int], *, iterations: int = 10_000) -> dict:
    result_path = run_dir / "result.json"
    ids_path = run_dir / "validation-ids.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") != "PASS_NATIVE_FULL_EXTENT_TRAIN":
        raise ValueError("source run is not a completed native full-extent fit")
    if result.get("sealed_audit_loaded") is not False:
        raise ValueError("source run sealed-audit status is not explicitly false")
    if result.get("step") != 32768 or result.get("best_step") != 4096:
        raise ValueError("unexpected source-run endpoint or selected checkpoint")
    ids_hash = result["artifacts"].get("validation-ids.json")
    if ids_hash != digest(ids_path):
        raise ValueError("validation ID hash differs from the source-run receipt")
    records = json.loads(ids_path.read_text(encoding="utf-8"))
    video_indices = [i for i, row in enumerate(records) if row["id"].startswith("video:")]
    video_records = [records[i] for i in video_indices]
    matches = [SCENE_ID.fullmatch(row["id"]) for row in video_records]
    if not matches or any(match is None for match in matches):
        raise ValueError("unexpected or missing CLEVRER scene identity in validation IDs")
    scene_ids = [match.group(1) for match in matches if match is not None]
    predictions: dict[int, np.ndarray] = {}
    metrics: dict[int, dict[str, np.ndarray]] = {}
    input_hashes = {"result.json": digest(result_path), "validation-ids.json": digest(ids_path)}
    for step in steps:
        name = f"validation-{step:04d}.npy"
        path = run_dir / name
        expected_hash = result["artifacts"].get(name)
        if expected_hash is None or digest(path) != expected_hash:
            raise ValueError(f"saved prediction hash mismatch for {name}")
        values = np.load(path, allow_pickle=False)
        if values.shape[0] != len(records):
            raise ValueError(f"saved prediction count mismatch for {name}")
        predictions[step] = values
        metrics[step] = per_example(values[video_indices], video_records)
        input_hashes[name] = expected_hash
    reference, endpoint = steps[0], steps[-1]
    reference_rows = metrics[reference]
    endpoint_rows = metrics[endpoint]
    transitions = {
        "correct_to_correct": int(
            np.sum((reference_rows["accuracy"] == 1) & (endpoint_rows["accuracy"] == 1))
        ),
        "correct_to_wrong": int(
            np.sum((reference_rows["accuracy"] == 1) & (endpoint_rows["accuracy"] == 0))
        ),
        "wrong_to_correct": int(
            np.sum((reference_rows["accuracy"] == 0) & (endpoint_rows["accuracy"] == 1))
        ),
        "wrong_to_wrong": int(
            np.sum((reference_rows["accuracy"] == 0) & (endpoint_rows["accuracy"] == 0))
        ),
        "prediction_changed": int(
            sum(
                int(
                    np.argmax(predictions[reference][idx, : len(row["option_ids"])])
                    != np.argmax(predictions[endpoint][idx, : len(row["option_ids"])])
                )
                for idx, row in zip(video_indices, video_records, strict=True)
            )
        ),
    }
    paired = {}
    for metric_name in ("accuracy", "confidence", "logit_span", "top_two_margin", "nll"):
        difference = endpoint_rows[metric_name] - reference_rows[metric_name]
        paired[metric_name] = paired_scene_bootstrap(
            difference, scene_ids, iterations=iterations, seed=17
        )
    return {
        "status": "PASS_READ_ONLY_VIDEO_LOGIT_DIAGNOSTIC",
        "scope": "same exposed development validation; no fitting or checkpoint selection",
        "source_run": {
            "run_dir": str(run_dir.resolve()),
            "source_commit": result.get("source_commit"),
            "config_sha256": result.get("config_sha256"),
            "run_result_sha256": input_hashes["result.json"],
            "validation_ids_sha256": input_hashes["validation-ids.json"],
            "best_step": result["best_step"],
            "final_step": result["step"],
            "sealed_audit_loaded": False,
        },
        "video_queries": len(video_records),
        "independent_scene_ids": len(set(scene_ids)),
        "step_metrics": {str(step): summarize(metrics[step]) for step in steps},
        "paired_endpoint_transitions": {
            "from_step": reference,
            "to_step": endpoint,
            **transitions,
        },
        "paired_scene_bootstrap": paired,
        "input_hashes": input_hashes,
        "model_loaded": False,
        "training": False,
        "calibration_fitted": False,
        "sealed_audit_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", default=[512, 4096, 32768])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=10_000)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or output.is_relative_to(Path(__file__).resolve().parents[1]):
        raise ValueError("write to a new external output path")
    if args.steps != sorted(set(args.steps)) or args.steps[0] != 512 or args.steps[-1] != 32768:
        raise ValueError("steps must be unique, increasing, and include 512 and 32768")
    result = analyze(args.run_dir.resolve(), args.steps, iterations=args.iterations)
    result["analysis_script_sha256"] = digest(Path(__file__))
    output.mkdir(parents=True)
    destination = output / "diagnostic.json"
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": result["status"],
                "video_queries": result["video_queries"],
                "scenes": result["independent_scene_ids"],
                "output": str(destination),
            }
        )
    )


if __name__ == "__main__":
    main()
