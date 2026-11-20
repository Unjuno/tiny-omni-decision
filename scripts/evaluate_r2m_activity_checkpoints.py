"""Reload and compare selected/final R2-M activity checkpoints without training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from train_r2m_activity_decision import (
    CLASSIFIER_MODES,
    build_vocabulary,
    evaluate,
    load_frozen_examples,
    make_model,
    train_sensor_normalization,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_predictions(path: Path) -> tuple[list[str], list[dict[str, float]]]:
    ids: list[str] = []
    probabilities: list[dict[str, float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        ids.append(str(row["sample_id"]))
        probabilities.append({key: float(value)
                              for key, value in row["probabilities"].items()})
    return ids, probabilities


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--pair-index-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.batch_size < 1:
        raise SystemExit("batch size must be positive")
    result_path = args.run_dir / "results.json"
    config_path = args.run_dir / "effective-config.json"
    if not result_path.is_file() or not config_path.is_file():
        raise SystemExit("run results/config are missing; refusing partial evaluation")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    declared_config_hash = config.pop("effective_config_sha256_excluding_self", None)
    config_payload = json.dumps(config, sort_keys=True, indent=2) + "\n"
    if hashlib.sha256(config_payload.encode()).hexdigest() != declared_config_hash:
        raise SystemExit("effective config hash mismatch")
    full_config_hash = sha256_file(config_path)
    canonical_config = dict(config)
    canonical_config["effective_config_sha256_excluding_self"] = declared_config_hash
    canonical_config_text = json.dumps(canonical_config, sort_keys=True, indent=2) + "\n"
    canonical_config_hash = hashlib.sha256(canonical_config_text.encode()).hexdigest()
    if result.get("effective_config_sha256") != canonical_config_hash:
        raise SystemExit("results refer to a different canonical effective config")
    if result.get("sealed_audit_loaded") is not False:
        raise SystemExit("run did not establish sealed-audit isolation")

    metadata_path = args.pair_index_dir / "allowed-trial-metadata.json"
    features, _ = load_frozen_examples(args.feature_cache, metadata_path)
    vocabulary, _, _ = build_vocabulary(features)
    rows = sorted(
        (row for row in features.values()
         if row["split"] == "validation" and row["activity"] in vocabulary),
        key=lambda row: str(row["sample_id"]),
    )
    train_rows = [row for row in features.values()
                  if row["split"] == "train" and row["activity"] in vocabulary]
    mean, std = train_sensor_normalization([
        row["sensor"] for row in train_rows
    ])
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    device = torch.device(device_name)
    if device.type == "cpu":
        torch.set_num_threads(1)
    report = {
        "schema": "r2m_activity_checkpoint_reload_comparison_v1",
        "effective_config_file_sha256_actual": full_config_hash,
        "effective_config_canonical_sha256": canonical_config_hash,
        "effective_config_line_ending_note": (
            "Windows write_text translated canonical LF to CRLF; the exact file hash "
            "is reported separately from the canonical UTF-8/LF serialization hash."
        ),
        "validation_examples": len(rows),
        "vocabulary": list(vocabulary),
        "sealed_audit_loaded": False,
        "device": str(device),
        "arms": [],
    }
    summaries = {row["mode"]: row for row in result["arms"]}
    if set(summaries) != set(CLASSIFIER_MODES):
        raise SystemExit("run is missing a required classifier arm")

    for mode in CLASSIFIER_MODES:
        model, scorer = make_model(mode, len(vocabulary), device)
        expected_summary = summaries[mode]
        arm_report = {"mode": mode, "checkpoints": {}}
        for checkpoint_name, filename in (
            ("selected", f"{mode}-best-calibration-checkpoint.pt"),
            ("final", f"{mode}-final-checkpoint.pt"),
        ):
            checkpoint_path = args.run_dir / filename
            expected_hash_key = ("best_checkpoint_sha256" if checkpoint_name == "selected"
                                 else "final_checkpoint_sha256")
            checkpoint_hash = sha256_file(checkpoint_path)
            if checkpoint_hash != expected_summary[expected_hash_key]:
                raise SystemExit(f"checkpoint hash mismatch: {checkpoint_path.name}")
            checkpoint = torch.load(
                checkpoint_path, map_location=device, weights_only=False
            )
            if (checkpoint.get("mode") != mode
                    or tuple(checkpoint.get("vocabulary", ())) != vocabulary):
                raise SystemExit(f"checkpoint metadata mismatch: {checkpoint_path.name}")
            model.load_state_dict(checkpoint["fusion"], strict=True)
            scorer.load_state_dict(checkpoint["scorer"], strict=True)
            metrics, probabilities, _ = evaluate(
                model, scorer, rows, vocabulary, mode, device, args.batch_size,
                sensor_normalization=(mean, std),
            )
            prediction_path = args.run_dir / f"{mode}-validation-predictions.jsonl"
            max_saved_delta = None
            if checkpoint_name == "selected":
                saved_ids, saved_rows = read_predictions(prediction_path)
                if saved_ids != [str(row["sample_id"]) for row in rows]:
                    raise SystemExit(f"selected prediction IDs differ: {mode}")
                saved = [[saved_rows[row][label] for label in vocabulary]
                         for row in range(len(saved_rows))]
                max_saved_delta = float(abs(
                    probabilities - torch.tensor(saved, dtype=torch.float64).numpy()
                ).max())
                if max_saved_delta > 1e-5:
                    raise SystemExit(f"selected checkpoint reload parity failed: {mode}")
            else:
                final_predictions_path = (
                    args.run_dir / f"{mode}-final-validation-predictions.jsonl"
                )
                with final_predictions_path.open(
                    "w", encoding="utf-8", newline="\n"
                ) as stream:
                    for index, row in enumerate(rows):
                        prediction = int(probabilities[index].argmax())
                        stream.write(json.dumps({
                            "sample_id": str(row["sample_id"]),
                            "participant_id": int(row["participant_id"]),
                            "trial_id": int(row["trial_id"]),
                            "target": str(row["activity"]),
                            "prediction": vocabulary[prediction],
                            "probabilities": {
                                label: float(probabilities[index, class_index])
                                for class_index, label in enumerate(vocabulary)
                            },
                        }, sort_keys=True) + "\n")
            arm_report["checkpoints"][checkpoint_name] = {
                "epoch": int(checkpoint["epoch"]),
                "sha256": checkpoint_hash,
                "metrics": metrics,
                "selected_saved_prediction_max_abs_delta": max_saved_delta,
            }
        report["arms"].append(arm_report)

    output_path = args.run_dir / "checkpoint-reload-comparison.json"
    if output_path.exists():
        raise SystemExit("refusing to overwrite an existing checkpoint reload report")
    output_path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({
        "status": "PASS",
        "report": str(output_path),
        "arms": {
            row["mode"]: {
                name: {
                    "epoch": data["epoch"],
                    "macro_accuracy": data["metrics"]["macro_accuracy"],
                    "macro_nll": data["metrics"]["macro_nll"],
                    "selected_prediction_max_abs_delta":
                        data["selected_saved_prediction_max_abs_delta"],
                }
                for name, data in row["checkpoints"].items()
            }
            for row in report["arms"]
        },
        "sealed_audit_loaded": False,
    }, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
