"""Verify saved multi-query readout reloads to the recorded validation predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import yaml

from scripts.train_frozen_audio_multiquery import (
    CandidateScorer,
    _encode_texts,
    _evaluate,
    build_query_specs,
    make_query_examples,
)
from scripts.train_frozen_audio_probe import LABELS, _rows, _validate_splits
from tiny_omni_decision.dataset import sha256_file


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def _hash_jsonl_order(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256("\n".join(row["id"] for row in rows).encode("utf-8"))
    return digest.hexdigest()


def main() -> None:
    args = _args()
    run_dir = args.run_dir.resolve()
    config_path = run_dir / "effective-config.yaml"
    report_path = run_dir / "run-report.json"
    best_path = run_dir / "best-readout.pt"
    expected_predictions_path = run_dir / "validation-multiquery-predictions.jsonl"
    for path in (config_path, report_path, best_path, expected_predictions_path):
        if not path.is_file():
            raise FileNotFoundError(f"required frozen run artifact missing: {path}")

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    original_report_bytes = report_path.read_bytes()
    report = json.loads(original_report_bytes)
    if report.get("status") != "complete_frozen_audio_multiquery_development_experiment":
        raise ValueError("run report is not a completed multi-query experiment")
    if report.get("sealed_audit_loaded") is not False or report.get(
        "test_split_loaded"
    ) is not False:
        raise ValueError("run report indicates forbidden test or sealed-audit data use")
    if report.get("model", {}).get("readout_state_dict_sha256") != sha256_file(best_path):
        raise ValueError("best readout checkpoint hash differs from training report")
    if report.get("validation_predictions_sha256") != sha256_file(expected_predictions_path):
        raise ValueError("saved validation prediction hash differs from training report")
    if sha256_file(config_path) != report.get("effective_config_sha256"):
        raise ValueError("effective config hash differs from training report")

    sample_dir = Path(config["source"]["sample_dir"]).resolve()
    cache_path = Path(config["source"]["feature_cache"]).resolve()
    train_rows = _rows(sample_dir / "train.jsonl")
    validation_rows = _rows(sample_dir / "validation.jsonl")
    _validate_splits(sample_dir, train_rows, validation_rows)
    if sha256_file(sample_dir / "validation.jsonl") != report["data_hashes"]["validation_manifest"]:
        raise ValueError("frozen validation manifest hash mismatch")
    if sha256_file(cache_path) != report["data_hashes"]["feature_cache"]:
        raise ValueError("frozen feature cache hash mismatch")
    cache = torch.load(cache_path, map_location="cpu", weights_only=True)
    if cache.get("validation_ids") != [row["id"] for row in validation_rows]:
        raise ValueError("cached validation feature IDs/order changed")
    validation_audio = cache["validation_features"].float()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    text_model_path = Path(config["encoder"]["text_model_path"]).resolve()
    query_texts = {
        f"Question: {spec['question']}"
        for label in LABELS
        for spec in build_query_specs(label)
    }
    candidate_texts = {
        f"Candidate answer: {option}."
        for label in LABELS
        for spec in build_query_specs(label)
        for option in spec["options"]
    }
    text_vectors, text_parameters = _encode_texts(
        sorted(query_texts | candidate_texts), text_model_path, device
    )
    if text_parameters != int(config["encoder"]["text_parameters"]):
        raise ValueError("reloaded text encoder parameter count mismatch")
    validation_examples = make_query_examples(validation_rows, validation_audio, text_vectors)

    input_size = int(validation_examples[0]["features"].shape[-1])
    scorer = CandidateScorer(input_size).to(device)
    state = torch.load(best_path, map_location=device, weights_only=True)
    scorer.load_state_dict(state, strict=True)
    metrics, predictions = _evaluate(
        scorer,
        validation_examples,
        device,
        int(config["readout"]["batch_queries"]),
    )
    expected = [
        json.loads(line)
        for line in expected_predictions_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if len(expected) != len(predictions) or len(predictions) != len(validation_examples):
        raise ValueError("reloaded prediction count differs from saved validation record")
    max_logit_delta = 0.0
    max_probability_delta = 0.0
    for saved, actual in zip(expected, predictions, strict=True):
        for field in ("sample_id", "question_type", "target_index", "prediction_index", "options"):
            if saved[field] != actual[field]:
                raise ValueError(f"reloaded prediction identity differs in field {field}")
        logit_delta = max(
            abs(float(left) - float(right))
            for left, right in zip(saved["logits"], actual["logits"], strict=True)
        )
        probability_delta = max(
            abs(float(left) - float(right))
            for left, right in zip(saved["probabilities"], actual["probabilities"], strict=True)
        )
        max_logit_delta = max(max_logit_delta, logit_delta)
        max_probability_delta = max(max_probability_delta, probability_delta)
    if max_logit_delta > 1e-6 or max_probability_delta > 1e-7:
        raise ValueError(
            "reloaded logits/probabilities differ from saved predictions: "
            f"logit={max_logit_delta}, probability={max_probability_delta}"
        )
    metric_groups = [
        (metrics["macro_question_type"], report["validation_metrics"]["macro_question_type"])
    ] + [
        (
            metrics["by_question_type"][name],
            report["validation_metrics"]["by_question_type"][name],
        )
        for name in metrics["by_question_type"]
    ]
    max_metric_delta = max(
        abs(float(actual[metric]) - float(expected_metric[metric]))
        for actual, expected_metric in metric_groups
        for metric in ("accuracy", "nll", "brier", "ece_15_bins")
    )
    if max_metric_delta > 1e-7:
        raise ValueError(f"reloaded validation metrics differ from original: {max_metric_delta}")

    verification = {
        "status": "passed",
        "verifier_sha256": sha256_file(Path(__file__).resolve()),
        "run_report_sha256": hashlib.sha256(original_report_bytes).hexdigest(),
        "best_checkpoint_sha256": sha256_file(best_path),
        "effective_config_sha256": sha256_file(config_path),
        "validation_manifest_sha256": sha256_file(sample_dir / "validation.jsonl"),
        "validation_ordered_ids_sha256": _hash_jsonl_order(validation_rows),
        "reloaded_prediction_count": len(predictions),
        "max_abs_logit_delta": max_logit_delta,
        "max_abs_probability_delta": max_probability_delta,
        "max_metric_delta": max_metric_delta,
        "reloaded_validation_metrics": metrics,
        "test_split_loaded": False,
        "sealed_audit_loaded": False,
        "retrained": False,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
    }
    verification_path = run_dir / "best-reload-verification.json"
    verification_path.write_text(
        json.dumps(verification, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if report_path.read_bytes() != original_report_bytes:
        raise AssertionError("verification unexpectedly modified the original run report")
    print(json.dumps(verification, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
