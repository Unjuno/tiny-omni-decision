"""One-time fixed-checkpoint evaluation on the frozen fresh-speaker cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tiny_omni_decision.grouped_eval_metrics import (
    classification_metrics,
    grouped_bootstrap,
    percentile_interval,
)
from tiny_omni_decision.textfree_audio_runtime import TextFreeAudioRuntime

CLASSES = ("yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go")
MANIFEST_SHA256 = "96c6fe86298ef234e3b49546df8e07858049283a1e1334e4982a5bdbad5daaa4"
READOUT_SHA256 = "171f6a9303b2fd9ad63efe2bee6af119549fb4f26ea663e9c1297f51a678e1c3"
TEMPERATURE = 0.42702477413437295
ENCODER_REVISION = "openai/whisper-tiny@169d4a4341b33bc18d8881c4b69c2e104e1cc0af"
FIT_CLASS_COUNTS = {
    "yes": 94,
    "no": 99,
    "up": 103,
    "down": 106,
    "left": 100,
    "right": 96,
    "on": 107,
    "off": 109,
    "stop": 101,
    "go": 101,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def metric_intervals(bootstrap: dict[str, list[float]]) -> dict[str, list[float]]:
    return {name: percentile_interval(values) for name, values in bootstrap.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--media-dir", type=Path, required=True)
    parser.add_argument("--readout", type=Path, required=True)
    parser.add_argument("--encoder-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        raise SystemExit("output directory must be empty")
    manifest_bytes = args.manifest.read_bytes()
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    if manifest_sha != MANIFEST_SHA256:
        raise SystemExit("frozen cohort manifest SHA-256 mismatch")
    if sha256_file(args.readout) != READOUT_SHA256:
        raise SystemExit("fixed calibrated readout SHA-256 mismatch")
    rows = read_jsonl(args.manifest)
    if len(rows) != 1000 or len({row["id"] for row in rows}) != 1000:
        raise SystemExit("frozen cohort must contain exactly 1,000 unique IDs")
    if {row["target"] for row in rows} != set(CLASSES):
        raise SystemExit("cohort label set differs from the frozen class order")
    group_ids = [row["speaker_group_sha256"] for row in rows]
    if len(set(group_ids)) != 562:
        raise SystemExit("cohort speaker-group count differs from its frozen preflight")

    media_rows = read_jsonl(args.media_dir / "media-inventory.jsonl")
    media_by_id = {row["id"]: row for row in media_rows}
    if len(media_by_id) != 1000 or set(media_by_id) != {row["id"] for row in rows}:
        raise SystemExit("materialized audio inventory does not exactly match frozen IDs")
    for row in rows:
        media = media_by_id[row["id"]]
        for key in ("source", "source_file", "source_row", "speaker_group_sha256", "target"):
            if media.get(key) != row.get(key):
                raise SystemExit(f"media inventory identity mismatch for {row['id']} ({key})")
        audio_path = args.media_dir / media["local_file"]
        if sha256_file(audio_path) != media["audio_sha256"]:
            raise SystemExit(f"audio SHA-256 mismatch for {row['id']}")

    torch.set_num_threads(4)
    load_started = time.perf_counter()
    runtime = TextFreeAudioRuntime(
        args.encoder_dir,
        args.readout,
        device=args.device,
        cache_entries=0,
        temperature=TEMPERATURE,
        expected_readout_sha256=READOUT_SHA256,
    )
    resident_parameter_count = runtime.parameter_count
    load_seconds = time.perf_counter() - load_started
    probabilities: list[list[float]] = []
    targets: list[int] = []
    predictions: list[dict[str, Any]] = []
    per_class_counts: Counter[str] = Counter()
    start = time.perf_counter()
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    options = [{"option_id": name, "class_id": name} for name in CLASSES]
    try:
        for index, row in enumerate(rows, start=1):
            media = media_by_id[row["id"]]
            audio_path = args.media_dir / media["local_file"]
            decision = runtime.decide(
                audio_path,
                options,
                expected_media_sha256=media["audio_sha256"],
            )
            option_probs = [float(item["probability"]) for item in decision["options"]]
            target = CLASSES.index(row["target"])
            prediction = int(max(range(len(option_probs)), key=option_probs.__getitem__))
            probabilities.append(option_probs)
            targets.append(target)
            per_class_counts[row["target"]] += 1
            predictions.append(
                {
                    "id": row["id"],
                    "speaker_group_sha256": row["speaker_group_sha256"],
                    "target": row["target"],
                    "prediction": CLASSES[prediction],
                    "probabilities": dict(zip(CLASSES, option_probs, strict=True)),
                    "confidence": option_probs[prediction],
                    "correct": prediction == target,
                    "audio_sha256": media["audio_sha256"],
                }
            )
            if index % 100 == 0:
                print(json.dumps({"evaluated": index, "total": len(rows)}), flush=True)
    finally:
        runtime.close()
    inference_seconds = time.perf_counter() - start

    uniform = [[1.0 / len(CLASSES)] * len(CLASSES) for _ in rows]
    majority_class = max(CLASSES, key=lambda label: FIT_CLASS_COUNTS[label])
    majority_index = CLASSES.index(majority_class)
    majority = [[float(index == majority_index) for index in range(len(CLASSES))] for _ in rows]
    primary_metrics = classification_metrics(probabilities, targets, ece_bins=15)
    uniform_metrics = classification_metrics(uniform, targets, ece_bins=15)
    majority_metrics = classification_metrics(majority, targets, ece_bins=15)
    bootstrap_by_baseline = {}
    for name, baseline in (("uniform_10", uniform), ("training_majority_one_hot", majority)):
        bootstrap = grouped_bootstrap(
            probabilities,
            targets,
            group_ids,
            replicates=10_000,
            seed=17,
            ece_bins=15,
            baseline_probabilities=baseline,
        )
        bootstrap_by_baseline[name] = {
            "candidate_metric_intervals": metric_intervals(bootstrap["metrics"]),
            "candidate_minus_baseline_nll_interval": percentile_interval(
                bootstrap["candidate_minus_baseline"]["nll"]
            ),
            "candidate_minus_baseline_brier_interval": percentile_interval(
                bootstrap["candidate_minus_baseline"]["brier"]
            ),
        }

    correct_by_class = Counter(
        row["target"] for row in predictions if row["correct"]
    )
    confusion = {actual: {predicted: 0 for predicted in CLASSES} for actual in CLASSES}
    for row in predictions:
        confusion[row["target"]][row["prediction"]] += 1
    per_class = {
        label: {
            "count": per_class_counts[label],
            "correct": correct_by_class[label],
            "accuracy_recall": correct_by_class[label] / per_class_counts[label],
        }
        for label in CLASSES
    }
    report = {
        "schema_version": 1,
        "status": "FRESH_SPEAKER_EVALUATION_COMPLETE",
        "claim_scope": (
            "one locked development transfer evaluation on previously unused official "
            "TRAIN speakers; narrow 10-way English keyword classification"
        ),
        "source": (
            "google/speech_commands@a751309c0fd613e8a5d30d77900f30e8b42bc2da/"
            "v0.02 official TRAIN only"
        ),
        "license": "CC-BY-4.0",
        "attribution": (
            "Pete Warden, Speech Commands: A Dataset for Limited-Vocabulary Speech "
            "Recognition (2018), https://arxiv.org/abs/1804.03209; Google Speech "
            "Commands v0.02, CC BY 4.0."
        ),
        "cohort_manifest_sha256": manifest_sha,
        "cohort_examples": len(rows),
        "unique_speaker_groups": len(set(group_ids)),
        "class_order": list(CLASSES),
        "candidate": {
            "checkpoint_sha256": READOUT_SHA256,
            "calibration_temperature": TEMPERATURE,
            "encoder_revision": ENCODER_REVISION,
            "encoder_parameters": 8_208_384,
            "readout_parameters": 3_850,
            "resident_parameters": resident_parameter_count,
            "training_or_selection_on_this_cohort": False,
        },
        "metrics": {
            "candidate": primary_metrics,
            "uniform_10_baseline": uniform_metrics,
            "training_majority_one_hot_baseline": {
                "class": majority_class,
                "fit_class_counts": FIT_CLASS_COUNTS,
                **majority_metrics,
            },
            "per_class": per_class,
            "confusion_matrix_actual_by_predicted": confusion,
            "paired_speaker_group_bootstrap": {
                "replicates": 10_000,
                "seed": 17,
                "percentile_interval": "two-sided 95%",
                "by_baseline": bootstrap_by_baseline,
            },
        },
        "runtime": {
            "device": args.device,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "load_seconds": load_seconds,
            "inference_seconds": inference_seconds,
            "examples_per_second": len(rows) / inference_seconds,
            "peak_cuda_memory_bytes": (
                torch.cuda.max_memory_allocated() if args.device == "cuda" else None
            ),
        },
        "safety": {
            "training_run": False,
            "official_validation_shards_read": False,
            "official_test_shards_read": False,
            "sealed_audit_opened": False,
            "readout_or_temperature_selection_on_cohort": False,
        },
    }
    predictions_path = args.output_dir / "predictions.jsonl"
    predictions_path.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in predictions), encoding="utf-8"
    )
    report["predictions_sha256"] = sha256_file(predictions_path)
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "metrics": primary_metrics,
        "majority_class": majority_class, "per_class_accuracy": per_class,
        "runtime": report["runtime"]}, indent=2))


if __name__ == "__main__":
    main()
