"""Recompute the fixed development validation through the raw-text runtime."""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from scripts.train_banking77_minilm_readout import (
    load_and_validate_data,
    metric_summary,
    read_jsonl,
    sha256_file,
)
from tiny_omni_decision.banking77_minilm_l6 import (
    Banking77MiniLML6Decision,
    class_ids,
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

    _, validation_rows, split_manifest = load_and_validate_data(args.data_dir)
    expected = read_jsonl(args.reference_predictions)
    if len(expected) != len(validation_rows):
        raise ValueError("Reference prediction count does not match validation rows")
    for row, reference in zip(validation_rows, expected, strict=True):
        if row["sample_id"] != reference["sample_id"]:
            raise ValueError("Validation ID/order differs from saved L6 run")

    started = datetime.now(UTC).isoformat()
    timer = time.perf_counter()
    runtime = Banking77MiniLML6Decision.from_local_files(args.model_dir, args.checkpoint)
    observed_probs: list[list[float]] = []
    with torch.inference_mode():
        for row in validation_rows:
            observed_probs.append(runtime.probabilities(row["text"]))
    elapsed = time.perf_counter() - timer

    reference_probs = torch.tensor([row["probabilities"] for row in expected])
    probabilities = torch.tensor(observed_probs)
    targets = torch.tensor([int(row["label"]) for row in validation_rows])
    predictions = probabilities.argmax(dim=-1)
    reference_predictions = torch.tensor([int(row["prediction"]) for row in expected])
    max_delta = float((probabilities - reference_probs).abs().max())
    exact_argmax = bool(torch.equal(predictions, reference_predictions))
    if not exact_argmax or max_delta > 2e-5:
        raise RuntimeError(
            f"Runtime parity failed: exact_argmax={exact_argmax}, max_delta={max_delta}"
        )

    logits = probabilities.clamp_min(1e-30).log()
    options = [
        {"option_id": f"intent-{index}", "class_id": class_id}
        for index, class_id in enumerate(class_ids())
    ]
    decision = runtime.decide(validation_rows[0]["text"], options)
    receipt: dict[str, Any] = {
        "format": "banking77_minilm_l6_raw_text_runtime_verification_v1",
        "started_at_utc": started,
        "ended_at_utc": datetime.now(UTC).isoformat(),
        "elapsed_seconds": elapsed,
        "device": "cpu",
        "validation_rows": len(validation_rows),
        "split_manifest_sha256": sha256_file(args.data_dir / "split-manifest.json"),
        "reference_predictions_sha256": sha256_file(args.reference_predictions),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "validation_accuracy": metric_summary(logits, targets)["accuracy"],
        "validation_metrics": metric_summary(logits, targets),
        "exact_argmax_matches_reference": exact_argmax,
        "max_probability_abs_delta_vs_saved_readout": max_delta,
        "sample_json_valid": (
            len(decision["options"]) == 77
            and decision["selected_option_id"].startswith("intent-")
        ),
        "option_contract": (
            "all 77 fixed BANKING77 class IDs; arbitrary text options unsupported"
        ),
        "official_test_loaded": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
