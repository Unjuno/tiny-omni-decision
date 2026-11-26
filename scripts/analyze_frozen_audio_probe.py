"""Recompute selected-checkpoint metrics from the frozen audio cache on CPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from scripts.train_frozen_audio_probe import (
    LABELS,
    _metrics,
    _option_order_stability,
    _question_features,
    _rows,
    _validate_splits,
)
from scripts.train_frozen_text_probe import CandidateScorer, _predict
from tiny_omni_decision.dataset import sha256_file


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _args()
    sample_dir = args.sample_dir.resolve()
    run_dir = args.run_dir.resolve()
    report_path = run_dir / "run-report.json"
    cache_path = run_dir / "audio-features.pt"
    readout_path = run_dir / "best-readout.pt"
    predictions_path = run_dir / "validation-predictions.jsonl"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    cache = torch.load(cache_path, map_location="cpu", weights_only=True)
    train_rows = _rows(sample_dir / "train.jsonl")
    validation_rows = _rows(sample_dir / "validation.jsonl")
    _validate_splits(sample_dir, train_rows, validation_rows)
    if cache["train_ids"] != [row["id"] for row in train_rows]:
        raise ValueError("cached train IDs differ from the fixed sample order")
    if cache["validation_ids"] != [row["id"] for row in validation_rows]:
        raise ValueError("cached validation IDs differ from the fixed sample order")
    if cache["train_targets"] != [LABELS.index(row["target"]) for row in train_rows]:
        raise ValueError("cached train labels differ from the fixed sample")
    if cache["validation_targets"] != [LABELS.index(row["target"]) for row in validation_rows]:
        raise ValueError("cached validation labels differ from the fixed sample")

    option_embeddings = cache["option_embeddings"]
    train_questions = _question_features(cache["train_features"], option_embeddings)
    validation_questions = _question_features(cache["validation_features"], option_embeddings)
    scorer = CandidateScorer(train_questions[0].shape[-1])
    scorer.load_state_dict(torch.load(readout_path, map_location="cpu", weights_only=True))
    train_logits = _predict(scorer, train_questions, torch.device("cpu"))
    validation_logits = _predict(scorer, validation_questions, torch.device("cpu"))
    train_targets = cache["train_targets"]
    validation_targets = cache["validation_targets"]
    saved_predictions = [
        json.loads(line)
        for line in predictions_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if [row["id"] for row in saved_predictions] != cache["validation_ids"]:
        raise ValueError("saved prediction IDs differ from validation order")
    for row, logits, target in zip(
        saved_predictions, validation_logits, validation_targets, strict=True
    ):
        probabilities = torch.softmax(logits.float(), dim=-1).tolist()
        if row["target"] != LABELS[target] or len(row["probabilities"]) != len(LABELS):
            raise ValueError(f"saved prediction target/options are inconsistent for {row['id']}")
        if (
            max(
                abs(float(left) - float(right))
                for left, right in zip(row["probabilities"], probabilities, strict=True)
            )
            > 1e-6
        ):
            raise ValueError(
                f"saved probability vector does not reload from best checkpoint: {row['id']}"
            )

    result: dict[str, Any] = {
        "schema_version": 1,
        "run_report_sha256": sha256_file(report_path),
        "audio_feature_cache_sha256": sha256_file(cache_path),
        "selected_readout_sha256": sha256_file(readout_path),
        "validation_predictions_sha256": sha256_file(predictions_path),
        "source_fetch_report_sha256": sha256_file(sample_dir / "fetch-report.json"),
        "train_count": len(train_targets),
        "validation_count": len(validation_targets),
        "train_metrics_recomputed_from_best_readout": _metrics(train_logits, train_targets),
        "validation_metrics_recomputed_from_best_readout": _metrics(
            validation_logits, validation_targets
        ),
        "candidate_order_stability_validation": _option_order_stability(
            scorer,
            cache["validation_features"],
            option_embeddings,
            validation_targets,
            torch.device("cpu"),
        ),
        "prediction_rows_probability_reloaded": len(saved_predictions),
        "selected_epoch": report["best_epoch"],
        "selection_rule": "minimum aggregate validation NLL over recorded epochs",
        "sealed_audit_loaded": False,
    }
    output = run_dir / "posthoc-metrics.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
