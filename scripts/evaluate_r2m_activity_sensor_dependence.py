"""Fixed-checkpoint sensor-dependence diagnostic; performs no training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from analyze_r2m_activity_comparison import (
    paired_bootstrap,
    read_jsonl,
    rows_to_arrays,
)
from train_r2m_activity_decision import (
    CLASSIFIER_MODES,
    build_vocabulary,
    evaluate,
    load_frozen_examples,
    make_model,
    prediction_rows,
    sha256_file,
)

from tiny_omni_decision.r2m_activity import (
    same_participant_different_activity_sensor_donors,
    train_sensor_normalization,
)


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_predictions(path: Path, rows, vocabulary, probabilities) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in prediction_rows(rows, vocabulary, probabilities):
            stream.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--pair-index-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    args = parser.parse_args()
    if args.batch_size < 1 or args.bootstrap_repetitions < 100:
        raise SystemExit("batch size must be positive and bootstrap repetitions at least 100")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit("refusing to overwrite an existing diagnostic output directory")

    result = read_json(args.run_dir / "results.json")
    config_path = args.run_dir / "effective-config.json"
    config = read_json(config_path)
    if (result.get("sealed_audit_loaded") is not False
            or config.get("sealed_audit_loaded") is not False):
        raise SystemExit("run does not prove sealed-audit isolation")
    if result.get("schema") != "r2m_supplied_activity_decision_results_v1":
        raise SystemExit("unsupported or unexpected run result schema")
    expected_arms = {str(row["mode"]): row for row in result["arms"]}
    if set(expected_arms) != set(CLASSIFIER_MODES):
        raise SystemExit("run is missing an expected classifier arm")

    config_for_hash = dict(config)
    declared_hash = config_for_hash.pop("effective_config_sha256_excluding_self", None)
    canonical_text = json.dumps(config_for_hash, sort_keys=True, indent=2) + "\n"
    canonical_hash = hashlib.sha256(canonical_text.encode("utf-8")).hexdigest()
    full_config_hash = sha256_file(config_path)
    if canonical_hash != declared_hash:
        raise SystemExit("effective config canonical hash mismatch")

    metadata_path = args.pair_index_dir / "allowed-trial-metadata.json"
    features, data_info = load_frozen_examples(args.feature_cache, metadata_path)
    vocabulary, _, _ = build_vocabulary(features)
    rows = sorted(
        (row for row in features.values()
         if row["split"] == "validation" and row["activity"] in vocabulary),
        key=lambda row: str(row["sample_id"]),
    )
    train_rows = [
        row for row in features.values()
        if row["split"] == "train" and row["activity"] in vocabulary
    ]
    mean, standard_deviation = train_sensor_normalization([
        np.asarray(row["sensor"], dtype=np.float32) for row in train_rows
    ])
    donors = same_participant_different_activity_sensor_donors(rows, seed=17)
    donor_ids = {str(row["sample_id"]) for row in donors}
    if donor_ids != {str(row["sample_id"]) for row in rows}:
        raise SystemExit("donor rows are not a permutation of validation rows")

    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    device = torch.device(device_name)
    if device.type == "cpu":
        torch.set_num_threads(1)

    mode = "audio_sensor_gated"
    checkpoint_path = args.run_dir / f"{mode}-best-calibration-checkpoint.pt"
    checkpoint_hash = sha256_file(checkpoint_path)
    if checkpoint_hash != expected_arms[mode]["best_checkpoint_sha256"]:
        raise SystemExit("selected checkpoint SHA-256 differs from the run results")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if (checkpoint.get("mode") != mode
            or tuple(checkpoint.get("vocabulary", ())) != vocabulary):
        raise SystemExit("selected checkpoint metadata mismatch")
    model, scorer = make_model(mode, len(vocabulary), device)
    model.load_state_dict(checkpoint["fusion"], strict=True)
    scorer.load_state_dict(checkpoint["scorer"], strict=True)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    original_metrics, original_probs, targets = evaluate(
        model, scorer, rows, vocabulary, mode, device, args.batch_size,
        sensor_normalization=(mean, standard_deviation),
    )
    saved_rows = read_jsonl(args.run_dir / f"{mode}-validation-predictions.jsonl")
    ids, saved_targets, saved_probs, participants = rows_to_arrays(saved_rows, vocabulary)
    if ids != [str(row["sample_id"]) for row in rows]:
        raise SystemExit("saved and current validation IDs differ")
    if not np.array_equal(targets, saved_targets):
        raise SystemExit("saved and current validation targets differ")
    reload_delta = float(np.max(np.abs(original_probs - saved_probs)))
    if reload_delta > 1e-5:
        raise SystemExit("fixed checkpoint does not reproduce saved validation predictions")

    mismatched_metrics, mismatched_probs, _ = evaluate(
        model, scorer, rows, vocabulary, mode, device, args.batch_size,
        sensor_replacement=donors,
        sensor_normalization=(mean, standard_deviation),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    original_path = args.output_dir / "original-selected-validation-predictions.jsonl"
    mismatched_path = args.output_dir / (
        "same-participant-different-activity-sensor-predictions.jsonl"
    )
    write_predictions(original_path, rows, vocabulary, original_probs)
    write_predictions(mismatched_path, rows, vocabulary, mismatched_probs)

    ids_hash = hashlib.sha256("\n".join(ids).encode()).hexdigest()
    bootstrap = paired_bootstrap(
        targets, original_probs, mismatched_probs, participants,
        vocabulary=vocabulary, repetitions=args.bootstrap_repetitions,
    )
    for metric in bootstrap.values():
        metric["mean_delta_mismatch_minus_original"] = metric.pop(
            "mean_delta_fused_minus_audio"
        )
    donor_activity_mismatches = sum(
        str(row["activity"]) != str(donor["activity"])
        for row, donor in zip(rows, donors, strict=True)
    )
    same_participant_pairs = sum(
        int(row["participant_id"]) == int(donor["participant_id"])
        for row, donor in zip(rows, donors, strict=True)
    )
    output = {
        "schema": "r2m_activity_sensor_dependence_diagnostic_v1",
        "purpose": (
            "fixed selected checkpoint input-dependence diagnostic; no training, "
            "checkpoint selection, or sealed-audit access"
        ),
        "run_config_sha256_actual": full_config_hash,
        "run_config_sha256_canonical": canonical_hash,
        "selected_checkpoint_sha256": checkpoint_hash,
        "selected_checkpoint_epoch": int(checkpoint["epoch"]),
        "analysis_source_sha256": sha256_file(Path(__file__).resolve()),
        "donor_helper_source_sha256": sha256_file(
            Path(__file__).resolve().parents[1] / "src" / "tiny_omni_decision"
            / "r2m_activity.py"
        ),
        "model_runner_source_sha256": sha256_file(
            Path(__file__).resolve().parents[1] / "scripts"
            / "train_r2m_activity_decision.py"
        ),
        "feature_cache_index_sha256": data_info["feature_cache_index_sha256"],
        "metadata_sha256": data_info["metadata_sha256"],
        "validation_ids_sha256": ids_hash,
        "validation_examples": len(rows),
        "participants": sorted(set(int(value) for value in participants.tolist())),
        "donor_constraints": {
            "mapping_is_permutation": donor_ids == {str(row["sample_id"]) for row in rows},
            "same_participant_pairs": same_participant_pairs,
            "different_activity_pairs": donor_activity_mismatches,
            "seed": 17,
            "construction": (
                "within each participant, sort rows by activity/sample ID; shuffle donor rows "
                "inside activity blocks; circularly shift the donor sequence by half the "
                "participant's rows. Fails if one activity occupies over half of a "
                "participant's rows."
            ),
        },
        "device": str(device),
        "peak_torch_cuda_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        ),
        "original": original_metrics,
        "mismatched_sensor": mismatched_metrics,
        "paired_mismatch_minus_original_bootstrap": {
            "method": (
                "paired validation-trial bootstrap stratified by participant × target activity"
            ),
            "repetitions": args.bootstrap_repetitions,
            "seed": 17,
            "interval": "percentile 95%",
            "metrics": bootstrap,
            "limitations": (
                "Only two held-out participant clusters are present; intervals are conditional "
                "on these two participants. This reuses development validation for a fixed "
                "checkpoint sensitivity analysis and is not blind audit or generalization evidence."
            ),
        },
        "argmax_changed_examples": int(np.sum(
            original_probs.argmax(axis=1) != mismatched_probs.argmax(axis=1)
        )),
        "selected_checkpoint_saved_prediction_max_abs_delta": reload_delta,
        "prediction_files": {
            "original_sha256": sha256_file(original_path),
            "mismatched_sensor_sha256": sha256_file(mismatched_path),
        },
        "sealed_audit_loaded": False,
        "training_performed": False,
        "checkpoint_modified": False,
    }
    report_path = args.output_dir / "diagnostic.json"
    report_path.write_text(json.dumps(output, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "report": str(report_path),
        "original_macro_accuracy": original_metrics["macro_accuracy"],
        "mismatched_macro_accuracy": mismatched_metrics["macro_accuracy"],
        "argmax_changed_examples": output["argmax_changed_examples"],
        "selected_checkpoint_saved_prediction_max_abs_delta": reload_delta,
        "sealed_audit_loaded": False,
        "training_performed": False,
    }, indent=2))


if __name__ == "__main__":
    main()
