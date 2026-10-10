"""Check end-to-end runtime parity on the saved Speech Commands validation IDs."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from scripts.train_frozen_audio_multiquery import (
    QUESTION_TASKS,
    _metrics,
    build_query_specs,
)
from scripts.train_frozen_audio_probe import _rows, _validate_splits
from tiny_omni_decision.audio_runtime import DecisionQuery, SpeechCommandsMultiQueryRuntime


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--baseline-run-report", type=Path, required=True)
    parser.add_argument("--baseline-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--audio-batch-size", type=int, default=8)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _verify_bundle(bundle: Path, manifest: dict[str, Any]) -> None:
    expected_paths = set()
    for entry in manifest.get("files", []):
        relative = Path(str(entry["path"]))
        path = (bundle / relative).resolve()
        if not path.is_relative_to(bundle.resolve()):
            raise ValueError("bundle manifest contains a path outside the bundle")
        if not path.is_file() or path.stat().st_size != int(entry["bytes"]):
            raise ValueError(f"bundle file missing or wrong-sized: {relative}")
        if _sha256(path) != entry["sha256"]:
            raise ValueError(f"bundle file hash mismatch: {relative}")
        expected_paths.add(relative.as_posix())
    actual_paths = {
        path.relative_to(bundle).as_posix()
        for path in bundle.rglob("*")
        if path.is_file() and path.name != "bundle-manifest.json"
    }
    if actual_paths != expected_paths:
        raise ValueError("bundle file set differs from its manifest")


def main() -> None:
    args = _args()
    bundle = args.bundle.resolve()
    sample_dir = args.sample_dir.resolve()
    baseline_report_path = args.baseline_run_report.resolve()
    baseline_predictions_path = args.baseline_predictions.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to reuse runtime parity output: {output}")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if args.audio_batch_size < 1:
        raise ValueError("--audio-batch-size must be positive")

    manifest = json.loads((bundle / "bundle-manifest.json").read_text(encoding="utf-8"))
    if manifest.get("bundle_id") != "path-a-audio-multiquery-runtime-v0":
        raise ValueError("unexpected runtime bundle ID")
    _verify_bundle(bundle, manifest)
    source_report = json.loads(baseline_report_path.read_text(encoding="utf-8"))
    if _sha256(baseline_predictions_path) != source_report["validation_predictions_sha256"]:
        raise ValueError("baseline prediction file does not match the frozen source report")
    rows = _rows(sample_dir / "validation.jsonl")
    validation_hash = _sha256(sample_dir / "validation.jsonl")
    if validation_hash != source_report["data_hashes"]["validation_manifest"]:
        raise ValueError("validation manifest hash differs from the frozen source run")
    baseline = _jsonl(baseline_predictions_path)
    if len(baseline) != len(rows) * len(QUESTION_TASKS):
        raise ValueError("baseline prediction count does not match validation rows and query types")
    _validate_splits(sample_dir, _rows(sample_dir / "train.jsonl"), rows)
    expected_baseline = []
    for row in rows:
        for spec in build_query_specs(str(row["target"])):
            expected_baseline.append(
                (str(row["id"]), spec["question_type"], spec["question"], spec["options"])
            )
    for actual, expected in zip(baseline, expected_baseline, strict=True):
        identity = (
            str(actual["sample_id"]),
            str(actual["question_type"]),
            str(actual["question"]),
            actual["options"],
        )
        if identity != expected:
            raise ValueError(
                "baseline query order/content differs from current frozen query template"
            )

    selected_rows = rows[: args.limit] if args.limit is not None else rows
    selected_baseline = baseline[: len(selected_rows) * len(QUESTION_TASKS)]
    output.mkdir(parents=True)
    started = datetime.now(UTC)
    load_started = time.perf_counter()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    runtime = SpeechCommandsMultiQueryRuntime(
        bundle / "models", bundle / "readout" / "best-readout.pt", device=device
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        load_peak = torch.cuda.max_memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
    else:
        load_peak = None
    load_seconds = time.perf_counter() - load_started

    metrics_logits: dict[str, list[torch.Tensor]] = defaultdict(list)
    metrics_targets: dict[str, list[int]] = defaultdict(list)
    predictions = []
    max_logit_delta = 0.0
    max_probability_delta = 0.0
    prediction_mismatches = 0
    inference_started = time.perf_counter()
    for batch_start in range(0, len(selected_rows), args.audio_batch_size):
        batch_rows = selected_rows[batch_start : batch_start + args.audio_batch_size]
        batch_specs = [build_query_specs(str(row["target"])) for row in batch_rows]
        batch_queries = [
            [
                DecisionQuery(question=str(spec["question"]), options=tuple(spec["options"]))
                for spec in specs
            ]
            for specs in batch_specs
        ]
        batch_results = runtime.decide_batch(
            [sample_dir / row["media_path"] for row in batch_rows], batch_queries
        )
        for local_index, (row, specs, results) in enumerate(
            zip(batch_rows, batch_specs, batch_results, strict=True)
        ):
            row_index = batch_start + local_index
            for task_index, (spec, result) in enumerate(zip(specs, results, strict=True)):
                reference = selected_baseline[row_index * len(QUESTION_TASKS) + task_index]
                logits = result.option_logits.float()
                probs = result.probabilities.float()
                target_index = int(spec["target_index"])
                ref_logits = torch.tensor(reference["logits"], dtype=torch.float32)
                ref_probs = torch.tensor(reference["probabilities"], dtype=torch.float32)
                max_logit_delta = max(max_logit_delta, float((logits - ref_logits).abs().max()))
                max_probability_delta = max(
                    max_probability_delta, float((probs - ref_probs).abs().max())
                )
                prediction_mismatches += int(
                    result.predicted_index != int(reference["prediction_index"])
                )
                metrics_logits[spec["question_type"]].append(logits)
                metrics_targets[spec["question_type"]].append(target_index)
                predictions.append(
                    {
                        "sample_id": str(row["id"]),
                        "media_sha256": str(row["media_sha256"]),
                        "question_type": spec["question_type"],
                        "question": spec["question"],
                        "options": spec["options"],
                        "target_index": target_index,
                        "prediction_index": result.predicted_index,
                        "logits": logits.tolist(),
                        "probabilities": probs.tolist(),
                        "reference_prediction_index": int(reference["prediction_index"]),
                    }
                )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        inference_peak = torch.cuda.max_memory_allocated(device)
    else:
        inference_peak = None
    inference_seconds = time.perf_counter() - inference_started

    by_type = {
        task: {
            **_metrics(metrics_logits[task], metrics_targets[task]),
            "query_count": len(metrics_targets[task]),
        }
        for task in QUESTION_TASKS
    }
    macro = {
        metric: sum(float(by_type[task][metric]) for task in QUESTION_TASKS)
        / len(QUESTION_TASKS)
        for metric in ("accuracy", "nll", "brier", "ece_15_bins")
    }
    predictions_path = output / "runtime-validation-predictions.jsonl"
    predictions_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in predictions),
        encoding="utf-8",
    )
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    ).stdout.strip()
    parity_passed = (
        prediction_mismatches == 0
        and max_logit_delta <= 1e-4
        and max_probability_delta <= 1e-5
    )
    report = {
        "schema_version": 1,
        "status": "passed_end_to_end_validation_parity" if parity_passed else "failed_parity",
        "scope": (
            "development validation replay for packaging parity; "
            "no model selection or calibration"
        ),
        "started_utc": started.isoformat(),
        "finished_utc": datetime.now(UTC).isoformat(),
        "source_commit": source_commit,
        "bundle_id": manifest["bundle_id"],
        "bundle_manifest_sha256": _sha256(bundle / "bundle-manifest.json"),
        "baseline_run_report_sha256": _sha256(baseline_report_path),
        "baseline_predictions_sha256": _sha256(baseline_predictions_path),
        "validation_manifest_sha256": _sha256(sample_dir / "validation.jsonl"),
        "validation_id_order_sha256": hashlib.sha256(
            "\n".join(str(row["id"]) for row in selected_rows).encode("utf-8")
        ).hexdigest(),
        "result_count": len(predictions),
        "unique_audio_observations": len(selected_rows),
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "runtime_parameters": runtime.parameter_count,
        "audio_encoder_batch_size": args.audio_batch_size,
        "temperature": 1.0,
        "training_performed": False,
        "test_split_loaded": False,
        "sealed_audit_loaded": False,
        "runtime_load_seconds": load_seconds,
        "end_to_end_validation_seconds": inference_seconds,
        "seconds_per_audio_observation": inference_seconds / len(selected_rows),
        "cuda_load_peak_allocated_bytes": load_peak,
        "cuda_inference_peak_allocated_bytes": inference_peak,
        "validation_metrics": {"by_question_type": by_type, "macro_question_type": macro},
        "parity": {
            "prediction_mismatches": prediction_mismatches,
            "max_logit_absolute_delta": max_logit_delta,
            "max_probability_absolute_delta": max_probability_delta,
            "acceptance": (
                "zero prediction mismatches, logit delta <=1e-4, "
                "probability delta <=1e-5"
            ),
        },
        "predictions": {
            "path": str(predictions_path),
            "sha256": _sha256(predictions_path),
            "bytes": predictions_path.stat().st_size,
        },
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not parity_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
