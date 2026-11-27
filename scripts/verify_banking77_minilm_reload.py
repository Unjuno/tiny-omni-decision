"""Verify a saved MiniLM readout reloads to the frozen validation predictions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from transformers import AutoModel, AutoTokenizer

from scripts.run_banking77_minilm_probe import (
    MODEL_REVISION,
    MODEL_WEIGHT_SHA256,
    _metrics,
    _pool_features,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_reload(
    train_csv: Path,
    split_json: Path,
    model_dir: Path,
    run_dir: Path,
    output_path: Path,
) -> dict[str, Any]:
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    split = json.loads(split_json.read_text(encoding="utf-8"))
    if sha256(train_csv) != result["data_sha256"]:
        raise ValueError("train data hash differs from the run receipt")
    if sha256(split_json) != result["split_manifest_sha256"]:
        raise ValueError("split manifest hash differs from the run receipt")
    model_dir = model_dir.resolve()
    if model_dir.name != MODEL_REVISION:
        raise ValueError("model snapshot revision mismatch")
    if sha256(model_dir / "model.safetensors") != MODEL_WEIGHT_SHA256:
        raise ValueError("pinned model weights hash mismatch")
    readout_path = run_dir / result["readout_file"]
    if sha256(readout_path) != result["readout_sha256"]:
        raise ValueError("saved readout hash differs from the run receipt")

    rows = list(
        csv.DictReader(
            io.StringIO(train_csv.read_text(encoding="utf-8-sig"), newline="")
        )
    )
    labels = sorted({row["category"] for row in rows})
    label_to_id = {label: index for index, label in enumerate(labels)}
    train_ids = split["train_row_ids"]
    validation_ids = split["validation_row_ids"]
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    encoder = AutoModel.from_pretrained(
        model_dir, local_files_only=True, use_safetensors=True
    ).eval().to("cpu")
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    features = _pool_features([row["text"] for row in rows], tokenizer, encoder, batch_size=64)
    targets = torch.tensor([label_to_id[row["category"]] for row in rows], dtype=torch.long)
    head = torch.nn.Linear(features.shape[-1], len(labels))
    head.load_state_dict(load_file(readout_path, device="cpu"))
    head.eval()
    with torch.no_grad():
        train_logits = head(features[train_ids])
        validation_logits = head(features[validation_ids])
    train_metrics = _metrics(train_logits, targets[train_ids], len(labels))
    validation_metrics = _metrics(
        validation_logits, targets[validation_ids], len(labels)
    )
    saved_predictions = json.loads(
        (run_dir / result["prediction_file"]).read_text(encoding="utf-8")
    )
    saved_probability = torch.tensor(
        [item["probabilities"] for item in saved_predictions]
    )
    reloaded_probability = validation_logits.softmax(-1)
    max_probability_delta = (
        saved_probability - reloaded_probability
    ).abs().max().item()
    prediction_matches = (
        reloaded_probability.argmax(-1).tolist()
        == [labels.index(item["predicted_intent"]) for item in saved_predictions]
    )
    if max_probability_delta > 1e-6 or not prediction_matches:
        raise ValueError("save/reload predictions do not match")

    def compact(metrics: dict[str, Any]) -> dict[str, Any]:
        omitted = {"predictions", "probabilities"}
        return {key: value for key, value in metrics.items() if key not in omitted}

    report = {
        "experiment_id": result["experiment_id"],
        "readout_sha256": sha256(readout_path),
        "prediction_sha256": sha256(run_dir / result["prediction_file"]),
        "reloaded_prediction_exact_match": prediction_matches,
        "max_probability_absolute_delta": max_probability_delta,
        "train_metrics": compact(train_metrics),
        "validation_metrics": compact(validation_metrics),
        "validation_metrics_match_run_receipt": all(
            abs(validation_metrics[key] - result["best_validation"][key]) < 1e-6
            for key in (
                "accuracy",
                "macro_accuracy",
                "nll",
                "macro_nll",
                "multiclass_brier",
                "macro_multiclass_brier",
                "top_label_ece_15_bins",
                "macro_classwise_ece",
                "mean_confidence",
            )
        ),
    }
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train_csv", type=Path)
    parser.add_argument("split_json", type=Path)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output_json", type=Path)
    args = parser.parse_args()
    report = verify_reload(
        args.train_csv, args.split_json, args.model_dir, args.run_dir, args.output_json
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
