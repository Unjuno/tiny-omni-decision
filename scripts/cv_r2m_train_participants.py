"""Participant-held-out CV using only the frozen R2-M training participants."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
import train_r2m_activity_decision as trainer

from tiny_omni_decision.r2m_participant_cv import leave_one_participant_out_plan


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, sort_keys=True, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    temporary.replace(path)


def _class_weights(rows: list[dict[str, object]], vocabulary: tuple[str, ...]) -> np.ndarray:
    counts = Counter(str(row["activity"]) for row in rows)
    if any(counts[label] == 0 for label in vocabulary):
        raise ValueError("every fixed activity choice must occur in each fit fold")
    values = np.asarray([
        len(rows) / (len(vocabulary) * counts[label]) for label in vocabulary
    ], dtype=np.float32)
    values /= values.mean()
    return values


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite existing output directory: {args.output_dir}")
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        raise SystemExit("epochs, batch size and learning rate must be positive")
    repository = Path(__file__).resolve().parents[1]
    split_path = repository / (
        "manifests/pretrained_reuse/"
        "r2m-audio-vibration-split-v2-20261120.json"
    )
    split_manifest = json.loads(split_path.read_text(encoding="utf-8"))
    if split_manifest.get("manifest_sha256") != trainer.SPLIT_SHA256:
        raise SystemExit("participant split differs from the frozen R2-M split v2")
    if split_manifest.get("sealed_audit_accessed") is not False:
        raise SystemExit("frozen split does not establish sealed-audit isolation")
    allowed_train_participants = set(map(int, split_manifest["splits"]["train"]))

    metadata_path = args.pair_index_dir / "allowed-trial-metadata.json"
    pair_index_path = args.pair_index_dir / "pair-index.json"
    pair_index = json.loads(pair_index_path.read_text(encoding="utf-8"))
    if (pair_index.get("manifest_sha256") != trainer.PAIR_INDEX_SHA256
            or pair_index.get("sealed_audit_accessed") is not False):
        raise SystemExit("pair index differs from the sealed-audit-safe frozen index")

    # The frozen loader verifies cache/index/media identities and rejects any sealed rows.
    features, data_summary = trainer.load_frozen_examples(args.feature_cache, metadata_path)
    train_rows = sorted((row for row in features.values()
                         if row["split"] == "train"
                         and int(row["participant_id"]) in allowed_train_participants),
                        key=lambda row: str(row["sample_id"]))
    observed_participants = {int(row["participant_id"]) for row in train_rows}
    if observed_participants != allowed_train_participants:
        raise SystemExit("frozen training feature participants do not match the split manifest")
    vocabulary = tuple(sorted({str(row["activity"]) for row in train_rows}))
    plan = leave_one_participant_out_plan(train_rows, seed=trainer.SEED)
    rows_by_id = {str(row["sample_id"]): row for row in train_rows}
    if len(rows_by_id) != len(train_rows):
        raise SystemExit("duplicate frozen training sample IDs")

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    if device.type == "cuda":
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(device)
        device_info = {
            "name": torch.cuda.get_device_name(device),
            "total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
        }
    else:
        device_info = {"name": "CPU"}

    effective_config: dict[str, object] = {
        "schema": "r2m_train_only_participant_group_cv_v1",
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repository, check=True,
            capture_output=True, text=True,
        ).stdout.strip(),
        "seed": trainer.SEED,
        "arms": ["audio_only", "audio_sensor_gated"],
        "selection": "minimum calibration macro NLL; calibration group is whole participant",
        "outer_validation": "leave-one-original-training-participant-out; pooled OOF once",
        "choices": list(vocabulary),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR over fixed epoch budget",
        "loss": "train-only inverse-frequency weighted cross entropy",
        "random_option_order": True,
        "early_stopping": False,
        "initialization": "fresh fold-specific model weights; no prior checkpoints loaded",
        "fusion_width": 96,
        "fusion_heads": 4,
        "outer_fold_plan": plan,
        "train_sample_count": len(train_rows),
        "unique_sample_count": len(rows_by_id),
        "unique_participant_count": len(observed_participants),
        "cache_integrity": data_summary,
        "split_manifest_sha256": sha256_file(split_path),
        "pair_index_sha256": sha256_file(pair_index_path),
        "metadata_sha256": sha256_file(metadata_path),
        "sealed_audit_loaded": False,
        "selection_or_evaluation_on_existing_calibration_validation": False,
        "device": str(device),
        "device_info": device_info,
        "python": sys.version,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_runtime": torch.version.cuda,
    }
    encoded = json.dumps(effective_config, sort_keys=True, indent=2) + "\n"
    effective_config["effective_config_sha256_excluding_self"] = hashlib.sha256(
        encoded.encode("utf-8")
    ).hexdigest()
    config_text = json.dumps(effective_config, sort_keys=True, indent=2) + "\n"

    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "effective-config.json").write_text(
        config_text, encoding="utf-8", newline="\n"
    )
    (args.output_dir / "effective-config.sha256").write_text(
        hashlib.sha256(config_text.encode("utf-8")).hexdigest() + "\n",
        encoding="ascii", newline="\n",
    )
    started_at = datetime.now(UTC).isoformat()
    run_status_path = args.output_dir / "run-status.json"
    write_json_atomic(run_status_path, {
        "status": "running", "started_at_utc": started_at,
        "completed_arms": [], "current_fold": 1,
    })
    oof: dict[str, dict[str, list[dict[str, object]]]] = {
        mode: {} for mode in ("audio_only", "audio_sensor_gated")
    }
    fold_summaries = []
    started = time.perf_counter()
    completed_arms = []
    for fold_index, fold in enumerate(plan, start=1):
        fit_rows = [rows_by_id[sample_id] for sample_id in fold["train_sample_ids"]]
        calibration_rows = [rows_by_id[sample_id]
                            for sample_id in fold["calibration_sample_ids"]]
        heldout_rows = [rows_by_id[sample_id]
                        for sample_id in fold["validation_sample_ids"]]
        sensor_norm = trainer.train_sensor_normalization([
            np.asarray(row["sensor"], dtype=np.float32) for row in fit_rows
        ])
        weights = _class_weights(fit_rows, vocabulary)
        fold_result: dict[str, object] = {
            "fold": fold_index,
            "outer_participant": fold["outer_participant"],
            "train_participants": fold["train_participants"],
            "calibration_participant": fold["calibration_participant"],
            "counts": {
                "train": len(fit_rows), "calibration": len(calibration_rows),
                "validation": len(heldout_rows),
                "unique_train_samples": len({str(r["sample_id"]) for r in fit_rows}),
            },
            "arms": {},
        }
        fold_dir = args.output_dir / (
            f"fold-{fold_index:02d}-participant-{fold['outer_participant']}"
        )
        fold_dir.mkdir()
        for mode in ("audio_only", "audio_sensor_gated"):
            arm_dir = fold_dir / mode
            arm_dir.mkdir()
            summary, state = trainer.train_arm(
                mode, fit_rows, calibration_rows, heldout_rows, vocabulary, weights,
                arm_dir, epochs=args.epochs, batch_size=args.batch_size,
                learning_rate=args.learning_rate, weight_decay=args.weight_decay,
                device=device, sensor_normalization=sensor_norm,
                validation_allow_missing_classes=True,
            )
            fusion, scorer, probabilities, labels = state
            del fusion, scorer, labels
            if len(probabilities) != len(heldout_rows):
                raise AssertionError("held-out prediction count differs from fold rows")
            per_row = trainer.prediction_rows(heldout_rows, vocabulary, probabilities)
            oof[mode][str(fold["outer_participant"])] = per_row
            summary["validation_participant"] = fold["outer_participant"]
            summary["validation_metric_classes_present"] = len({
                str(row["activity"]) for row in heldout_rows
            })
            fold_result["arms"][mode] = summary
            completed_arms.append({"fold": fold_index, "participant": fold["outer_participant"],
                                   "arm": mode, "selected_epoch": summary["selected_epoch"]})
            write_json_atomic(args.output_dir / "fold-results.json",
                              fold_summaries + [fold_result])
            write_json_atomic(run_status_path, {
                "status": "running", "started_at_utc": started_at,
                "completed_arms": completed_arms,
                "current_fold": fold_index,
                "next_arm": (
                    "audio_sensor_gated" if mode == "audio_only"
                    else "audio_only" if fold_index < len(plan) else None
                ),
                "next_fold": fold_index if mode == "audio_only" else fold_index + 1,
                "elapsed_seconds": time.perf_counter() - started,
            })
            if device.type == "cuda":
                torch.cuda.empty_cache()
        fold_summaries.append(fold_result)

    aggregate: dict[str, object] = {}
    pooled_predictions: dict[str, list[dict[str, object]]] = {}
    for mode in oof:
        records = [record for participant in sorted(oof[mode], key=int)
                   for record in oof[mode][participant]]
        records.sort(key=lambda row: str(row["sample_id"]))
        ids = [str(row["sample_id"]) for row in records]
        if len(ids) != len(train_rows) or len(set(ids)) != len(ids):
            raise AssertionError("OOF predictions do not cover every train sample exactly once")
        probabilities = np.asarray([
            [row["probabilities"][label] for label in vocabulary] for row in records
        ], dtype=np.float64)
        targets = np.asarray([vocabulary.index(str(row["target"])) for row in records])
        aggregate[mode] = trainer.activity_metrics(probabilities, targets, vocabulary)
        pooled_predictions[mode] = records
        prediction_path = args.output_dir / f"{mode}-oof-predictions.jsonl"
        prediction_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
            encoding="utf-8", newline="\n",
        )

    audio_by_id = {row["sample_id"]: row for row in pooled_predictions["audio_only"]}
    fusion_by_id = {row["sample_id"]: row for row in pooled_predictions["audio_sensor_gated"]}
    if audio_by_id.keys() != fusion_by_id.keys():
        raise AssertionError("paired OOF arms predict different sample IDs")
    participant_deltas = {}
    for participant in sorted(oof["audio_only"], key=int):
        audio_rows = oof["audio_only"][participant]
        fusion_rows = oof["audio_sensor_gated"][participant]
        audio_acc = np.mean([row["target"] == row["prediction"] for row in audio_rows])
        fusion_acc = np.mean([row["target"] == row["prediction"] for row in fusion_rows])
        participant_deltas[participant] = {
            "examples": len(audio_rows),
            "audio_accuracy": float(audio_acc),
            "audio_sensor_accuracy": float(fusion_acc),
            "fusion_minus_audio_accuracy_pp": float(100 * (fusion_acc - audio_acc)),
        }
    report = {
        "status": "PASS",
        "effective_config": effective_config,
        "folds": fold_summaries,
        "pooled_oof_metrics": aggregate,
        "participant_metrics": participant_deltas,
        "sealed_audit_loaded": False,
        "existing_calibration_validation_used": False,
        "interpretation": (
            "Exploratory seven-participant train-only grouped CV. OOF rows are "
            "paired by sample, but folds share training participants; metrics do "
            "not establish broad participant-population performance."
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_torch_cuda_bytes": int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda" else 0,
    }
    report_path = args.output_dir / "results.json"
    report_path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n",
                           encoding="utf-8", newline="\n")
    write_json_atomic(run_status_path, {
        "status": "complete", "started_at_utc": started_at,
        "finished_at_utc": datetime.now(UTC).isoformat(),
        "completed_arms": completed_arms,
        "elapsed_seconds": report["elapsed_seconds"],
    })
    (args.output_dir / "integrity-hashes.json").write_text(json.dumps({
        path.name: sha256_file(path)
        for path in (args.output_dir / "effective-config.json",
                     args.output_dir / "fold-results.json", report_path,
                     args.output_dir / "audio_only-oof-predictions.jsonl",
                     args.output_dir / "audio_sensor_gated-oof-predictions.jsonl")
    }, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--pair-index-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    try:
        result = run(args)
    except BaseException as error:
        status_path = args.output_dir / "run-status.json"
        if status_path.exists():
            try:
                status = json.loads(status_path.read_text(encoding="utf-8"))
                status.update({
                    "status": "failed", "error_type": type(error).__name__,
                    "error": str(error),
                    "finished_at_utc": datetime.now(UTC).isoformat(),
                })
                write_json_atomic(status_path, status)
            except OSError:
                pass
        raise
    print(json.dumps({"status": result["status"], "output_dir": str(args.output_dir),
                      "elapsed_seconds": result["elapsed_seconds"]}, sort_keys=True))


if __name__ == "__main__":
    main()
