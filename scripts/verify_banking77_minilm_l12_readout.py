"""Reload and replay the selected MiniLM-L12 readout on fixed validation rows."""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn

from scripts.train_banking77_minilm_readout import (
    CLASS_COUNT,
    extract_features,
    load_and_validate_data,
    metric_summary,
    read_jsonl,
    sha256_file,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite verification receipt: {args.output}")

    _, validation_rows, _ = load_and_validate_data(args.data_dir)
    expected = read_jsonl(args.reference_predictions)
    if len(expected) != len(validation_rows):
        raise ValueError("Reference prediction count does not match validation rows")
    for row, reference in zip(validation_rows, expected, strict=True):
        if row["sample_id"] != reference["sample_id"]:
            raise ValueError("Validation ID/order differs from saved L12 run")

    started = datetime.now(UTC).isoformat()
    timer = time.perf_counter()
    features = extract_features(validation_rows, args.model_dir)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("class_count") != CLASS_COUNT or checkpoint.get("best_epoch") != 6:
        raise ValueError("Unexpected selected L12 readout metadata")
    readout = nn.Linear(features.shape[-1], CLASS_COUNT).cpu().eval()
    readout.load_state_dict(checkpoint["state_dict"], strict=True)
    with torch.inference_mode():
        logits = readout((features - checkpoint["train_mean"]) / checkpoint["train_std"])
        probabilities = torch.softmax(logits, dim=-1)
        predictions = logits.argmax(dim=-1)
    elapsed = time.perf_counter() - timer

    reference_probs = torch.tensor([row["probabilities"] for row in expected])
    reference_predictions = torch.tensor([int(row["prediction"]) for row in expected])
    targets = torch.tensor([int(row["label"]) for row in validation_rows])
    max_delta = float((probabilities - reference_probs).abs().max())
    exact_argmax = bool(torch.equal(predictions, reference_predictions))
    if not exact_argmax or max_delta > 1e-5:
        raise RuntimeError(
            f"Reload parity failed: exact_argmax={exact_argmax}, max_delta={max_delta}"
        )

    receipt: dict[str, Any] = {
        "format": "banking77_minilm_l12_readout_reload_verification_v1",
        "started_at_utc": started,
        "ended_at_utc": datetime.now(UTC).isoformat(),
        "elapsed_seconds": elapsed,
        "device": "cpu",
        "validation_rows": len(validation_rows),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "reference_predictions_sha256": sha256_file(args.reference_predictions),
        "exact_argmax_matches_saved_predictions": exact_argmax,
        "max_probability_abs_delta": max_delta,
        "validation_metrics": metric_summary(logits, targets),
        "official_test_loaded": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
