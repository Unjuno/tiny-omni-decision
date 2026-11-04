"""Replay a small fixed slice to measure VideoPrism bundle inference parity."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import psutil

from tiny_omni_decision.bundle_validation import verify_bundle


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--baseline-predictions", type=Path, required=True)
    parser.add_argument("--baseline-run-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--images", type=int, default=8)
    args = parser.parse_args()

    bundle = args.bundle.resolve(strict=True)
    sample_dir = args.sample_dir.resolve(strict=True)
    source_root = args.source_root.resolve(strict=True)
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to reuse runtime validation output: {output}")
    if args.images < 1:
        raise ValueError("--images must be positive")
    manifest = json.loads((bundle / "bundle-manifest.json").read_text(encoding="utf-8"))
    verify_bundle(bundle, manifest)
    sys.path.insert(0, str(bundle / "runtime"))
    from tiny_omni_decision.videoprism_runtime import VideoPrismClevr4Runtime

    report_path = args.baseline_run_report.resolve(strict=True)
    source_report = json.loads(report_path.read_text(encoding="utf-8"))
    predictions_path = args.baseline_predictions.resolve(strict=True)
    if sha256_file(predictions_path) != source_report["checkpoint_selection_predictions_sha256"]:
        raise ValueError("baseline predictions do not match the source run report")
    baseline = jsonl(predictions_path)
    by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in baseline:
        by_image[str(item["image_id"])].append(item)
    selected_ids = list(by_image)[: args.images]
    if len(selected_ids) != args.images:
        raise ValueError("baseline prediction set is smaller than requested image count")

    process = psutil.Process()
    rss_before = process.memory_info().rss
    load_started = time.perf_counter()
    runtime = VideoPrismClevr4Runtime(bundle, source_root)
    load_seconds = time.perf_counter() - load_started
    rss_after_load = process.memory_info().rss
    predictions = []
    max_probability_delta = 0.0
    mismatches = 0
    inference_started = time.perf_counter()
    for start in range(0, len(selected_ids), 8):
        batch_ids = selected_ids[start : start + 8]
        image_paths = [
            sample_dir / "val" / "images" / f"{image_id}.png" for image_id in batch_ids
        ]
        encoded_features = runtime.encode_images(image_paths)
        for image_id, features in zip(batch_ids, encoded_features, strict=True):
            tasks = sorted(by_image[image_id], key=lambda item: item["taxonomy"])
            for reference in tasks:
                taxonomy = str(reference["taxonomy"])
                options = tuple(str(option) for option in reference["options"])
                result = runtime.decide_from_features(features, taxonomy, options)
                ref_probabilities = np.asarray(reference["probabilities"], dtype=np.float32)
                probability_delta = float(
                    np.max(np.abs(result.probabilities - ref_probabilities))
                )
                max_probability_delta = max(max_probability_delta, probability_delta)
                prediction_index = int(np.argmax(result.probabilities))
                reference_prediction_index = options.index(str(reference["prediction"]))
                mismatches += int(prediction_index != reference_prediction_index)
                predictions.append(
                    {
                        "image_id": image_id,
                        "taxonomy": taxonomy,
                        "target": reference["target"],
                        "options": list(options),
                        "logits": result.logits.tolist(),
                        "probabilities": result.probabilities.tolist(),
                        "reference_prediction_index": reference_prediction_index,
                        "prediction_index": prediction_index,
                    }
                )
    inference_seconds = time.perf_counter() - inference_started
    output.mkdir(parents=True)
    predictions_path = output / "runtime-predictions.jsonl"
    predictions_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in predictions),
        encoding="utf-8",
    )
    passed = mismatches == 0 and max_probability_delta <= 1e-5
    result = {
        "status": "passed_parity" if passed else "failed_parity",
        "scope": (
            "runtime packaging parity on existing development validation; no training or selection"
        ),
        "bundle_id": manifest["bundle_id"],
        "bundle_manifest_sha256": sha256_file(bundle / "bundle-manifest.json"),
        "source_run_report_sha256": sha256_file(report_path),
        "baseline_predictions_sha256": sha256_file(args.baseline_predictions.resolve()),
        "validation_image_count": len(selected_ids),
        "decision_count": len(predictions),
        "unique_image_ids_sha256": hashlib.sha256("\n".join(selected_ids).encode()).hexdigest(),
        "runtime_encoder_backend": "JAX CPU",
        "runtime_environment": {
            "python": sys.version,
            "jax": runtime.jax.__version__,
            "tensorflow": runtime.tf.__version__,
            "numpy": np.__version__,
            "devices": [str(device) for device in runtime.jax.devices()],
        },
        "runtime_load_seconds": load_seconds,
        "end_to_end_inference_seconds": inference_seconds,
        "seconds_per_image": inference_seconds / len(selected_ids),
        "rss_before_load_bytes": rss_before,
        "rss_after_load_bytes": rss_after_load,
        "peak_working_set_bytes": getattr(process.memory_info(), "peak_wset", None),
        "max_probability_absolute_delta": max_probability_delta,
        "prediction_mismatches": mismatches,
        "parity_passed": passed,
        "training_performed": False,
        "sealed_audit_loaded": False,
        "predictions_sha256": sha256_file(predictions_path),
    }
    (output / "report.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
