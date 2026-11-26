"""Check a frozen typed-text runtime against its saved validation decisions."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any

import torch

from scripts.train_frozen_text_probe import _load_examples
from tiny_omni_decision.dataset import sha256_file
from tiny_omni_decision.io import load_structured_file
from tiny_omni_decision.schema import DatasetManifest
from tiny_omni_decision.text_runtime import TypedTextDecisionRuntime


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--readout", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--baseline-run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--limit", type=int, default=8)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite text runtime verification: {output}")
    if args.limit < 1:
        raise ValueError("--limit must be positive")

    validation = args.validation.resolve(strict=True)
    run_dir = args.baseline_run_dir.resolve(strict=True)
    source_report = json.loads((run_dir / "run-report.json").read_text(encoding="utf-8"))
    baseline_path = run_dir / str(source_report["predictions_path"])
    posthoc = json.loads((run_dir / "posthoc-metrics.json").read_text(encoding="utf-8"))
    manifest_path = args.validation_manifest.resolve(strict=True)
    dataset_manifest = DatasetManifest.model_validate(load_structured_file(manifest_path))
    if source_report.get("status") != "complete_text_only_synthetic_probe":
        raise ValueError("baseline is not the pinned typed-text development probe")
    if sha256_file(validation) != source_report["validation_file_sha256"]:
        raise ValueError("validation file hash differs from the source run")
    if sha256_file(run_dir / "best-readout.pt") != posthoc["best_readout_sha256"]:
        raise ValueError("readout hash differs from the checkpoint-selection record")

    examples = _load_examples(validation, dataset_manifest)
    baseline = _jsonl(baseline_path)
    if len(examples) != len(baseline) or len(examples) != int(source_report["validation_examples"]):
        raise ValueError("validation examples and saved predictions differ in length")
    if [example.id for example in examples] != [str(row["id"]) for row in baseline]:
        raise ValueError("validation decision order differs from saved predictions")
    limit = min(args.limit, len(examples))

    started = time.perf_counter()
    runtime = TypedTextDecisionRuntime(
        args.model_dir.resolve(strict=True), args.readout.resolve(strict=True), device=args.device
    )
    load_seconds = time.perf_counter() - started
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    inference_started = time.perf_counter()
    rows = []
    mismatches = 0
    max_probability_delta = 0.0
    for example, reference in zip(examples[:limit], baseline[:limit], strict=True):
        result = runtime.decide(example.state, example.question, example.options)
        reference_probabilities = torch.tensor(reference["probabilities"], dtype=torch.float32)
        delta = float((result.probabilities - reference_probabilities).abs().max())
        max_probability_delta = max(max_probability_delta, delta)
        mismatches += int(example.options[result.predicted_index] != reference["prediction"])
        rows.append(
            {
                "id": example.id,
                "source_record_id": example.source_record_id,
                "target": example.target,
                "prediction": example.options[result.predicted_index],
                "options": list(example.options),
                "probabilities": result.probabilities.tolist(),
                "reference_probabilities": reference["probabilities"],
                "max_probability_delta": delta,
            }
        )
    inference_seconds = time.perf_counter() - inference_started
    output.mkdir(parents=True)
    predictions_path = output / "runtime-predictions.jsonl"
    predictions_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    report = {
        "status": "passed_parity"
        if mismatches == 0 and max_probability_delta <= 1e-5
        else "failed_parity",
        "scope": "existing synthetic development validation replay; no selection or calibration",
        "source_run_report_sha256": sha256_file(run_dir / "run-report.json"),
        "source_readout_sha256": sha256_file(args.readout.resolve()),
        "validation_manifest_sha256": sha256_file(validation),
        "validation_id_order_sha256": hashlib.sha256(
            "\n".join(example.id for example in examples[:limit]).encode("utf-8")
        ).hexdigest(),
        "decision_count": len(rows),
        "device": str(args.device),
        "runtime_parameters": runtime.parameter_count,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "runtime_load_seconds": load_seconds,
        "inference_seconds": inference_seconds,
        "seconds_per_decision": inference_seconds / len(rows),
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated()
        if args.device == "cuda"
        else None,
        "prediction_mismatches": mismatches,
        "max_probability_absolute_delta": max_probability_delta,
        "training_performed": False,
        "sealed_audit_loaded": False,
        "predictions_sha256": sha256_file(predictions_path),
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["status"] != "passed_parity":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
