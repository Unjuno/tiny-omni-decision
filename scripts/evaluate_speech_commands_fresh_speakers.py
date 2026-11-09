"""One-time fixed-checkpoint evaluation on the frozen fresh-speaker cohort."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
import wave
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
from tiny_omni_decision.speech_commands_labels import validate_speech_commands_target
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


def normalized_pcm_sha256(path: Path) -> str:
    with wave.open(str(path), "rb") as stream:
        if (
            stream.getnchannels() != 1
            or stream.getsampwidth() != 2
            or stream.getframerate() != 16_000
            or stream.getcomptype() != "NONE"
        ):
            raise ValueError(f"unexpected WAV format: {path}")
        return hashlib.sha256(stream.readframes(stream.getnframes())).hexdigest()


def metric_intervals(bootstrap: dict[str, list[float]]) -> dict[str, list[float]]:
    return {name: percentile_interval(values) for name, values in bootstrap.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--historical-exposure-dir", type=Path, required=True)
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
    try:
        for row in rows:
            validate_speech_commands_target(
                row["label"], f"{row['source_path_class']}/", row["target"]
            )
    except (KeyError, ValueError) as error:
        raise SystemExit(f"invalid pinned Speech Commands label mapping: {error}") from error
    group_ids = [row["speaker_group_sha256"] for row in rows]
    if len(set(group_ids)) != 562:
        raise SystemExit("cohort speaker-group count differs from its frozen preflight")

    media_rows = read_jsonl(args.media_dir / "media-inventory.jsonl")
    media_by_id = {row["id"]: row for row in media_rows}
    if len(media_by_id) != 1000 or set(media_by_id) != {row["id"] for row in rows}:
        raise SystemExit("materialized audio inventory does not exactly match frozen IDs")
    historical_rows = []
    for split in ("train", "validation"):
        historical = read_jsonl(args.historical_exposure_dir / f"{split}.jsonl")
        if any(row.get("split") != split for row in historical):
            raise SystemExit(f"historical {split} manifest has a split mismatch")
        historical_rows.extend(historical)
    historical_speakers = {row["speaker_group_sha256"] for row in historical_rows}
    historical_assets = {row["source_path_sha256"] for row in historical_rows}
    historical_records = {(row["source_file"], row["source_row"]) for row in historical_rows}
    historical_media_hashes = {row["media_sha256"] for row in historical_rows}
    selected_assets = {row["source_path_sha256"] for row in rows}
    selected_records = {(row["source_file"], row["source_row"]) for row in rows}
    selected_media_hashes = {row["audio_sha256"] for row in media_rows}
    if historical_speakers.intersection(group_ids):
        raise SystemExit("fresh cohort overlaps prior train/validation speakers")
    if historical_assets.intersection(selected_assets):
        raise SystemExit("fresh cohort overlaps prior train/validation source assets")
    if historical_records.intersection(selected_records):
        raise SystemExit("fresh cohort overlaps prior train/validation source rows")
    if historical_media_hashes.intersection(selected_media_hashes):
        raise SystemExit("fresh cohort has an exact WAV content hash in prior exposure")
    pcm_hashes = []
    for row in rows:
        media = media_by_id[row["id"]]
        for key in (
            "source",
            "source_file",
            "source_row",
            "speaker_group_sha256",
            "label",
            "target",
            "source_path_class",
        ):
            if media.get(key) != row.get(key):
                raise SystemExit(f"media inventory identity mismatch for {row['id']} ({key})")
        audio_path = args.media_dir / media["local_file"]
        if sha256_file(audio_path) != media["audio_sha256"]:
            raise SystemExit(f"audio SHA-256 mismatch for {row['id']}")
        pcm_hashes.append(normalized_pcm_sha256(audio_path))
    class_counts = Counter(row["target"] for row in rows)
    per_speaker = Counter(group_ids)
    if set(class_counts.values()) != {100} or max(per_speaker.values()) > 5:
        raise SystemExit("fresh cohort balance or per-speaker cap differs from frozen protocol")
    if len(selected_assets) != len(rows) or len(selected_records) != len(rows):
        raise SystemExit("fresh cohort repeats a source asset or source record")

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
    repository_root = Path(__file__).resolve().parents[1]
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository_root, text=True
    ).strip()
    config_path = (
        repository_root / "configs/pretrained_reuse/speech_commands_fresh_speakers_eval.json"
    )
    materialization_report_path = args.media_dir / "materialization-report.json"
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
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "source_commit": source_commit,
        "evaluator_script_sha256": sha256_file(Path(__file__).resolve()),
        "materializer_script_sha256": sha256_file(
            repository_root / "scripts/fetch_speech_commands_fresh_speakers.mjs"
        ),
        "media_inventory_sha256": sha256_file(args.media_dir / "media-inventory.jsonl"),
        "materialization_report_sha256": sha256_file(materialization_report_path),
        "cohort_examples": len(rows),
        "unique_speaker_groups": len(set(group_ids)),
        "identity_integrity": {
            "prior_train_validation_examples": len(historical_rows),
            "speaker_overlap": 0,
            "source_asset_path_overlap": 0,
            "source_record_overlap": 0,
            "exact_wav_sha256_overlap": 0,
            "selected_cohort_unique_pcm_contents": len(set(pcm_hashes)),
            "selected_cohort_pcm_duplicates": len(pcm_hashes) - len(set(pcm_hashes)),
            "historical_normalized_pcm_overlap": (
                "UNKNOWN: prior raw WAVs are absent; exact historical WAV hashes were compared"
            ),
        },
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
            "transformers": importlib.metadata.version("transformers"),
            "safetensors": importlib.metadata.version("safetensors"),
            "cuda": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
            "load_seconds": load_seconds,
            "inference_seconds": inference_seconds,
            "examples_per_second": len(rows) / inference_seconds,
            "peak_cuda_memory_bytes": (
                torch.cuda.max_memory_allocated() if args.device == "cuda" else None
            ),
        },
        "command": [sys.executable, *sys.argv],
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
