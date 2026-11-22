"""Replay a saved MiniWoB readout through the public Decision JSON contract.

This is an inference/serialization parity check. It never trains, selects, or
overwrites artifacts in the frozen experiment directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.train_miniwob_dino_path_a import read_corpus, targets_for  # noqa: E402
from tiny_omni_decision.miniwob_dino_readout import MiniWoBOptionScorer  # noqa: E402

PROBABILITY_TOLERANCE = 5e-6


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve(strict=True)
    output_dir = args.output_dir.resolve()
    if output_dir == run_dir or run_dir in output_dir.parents:
        raise ValueError("replay output must not be inside the frozen run directory")
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = run_dir / "effective-config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("cuda_used") is not False or config.get("device") != "cpu":
        raise ValueError("expected the recorded CPU-only run")
    _, _, splits = read_corpus(config)
    rows = splits["validation"]
    targets = targets_for(rows)
    cache_path = run_dir / "frozen-features.npz"
    with np.load(cache_path, allow_pickle=False) as archive:
        features = torch.from_numpy(archive["validation"].copy())
        cached_targets = torch.from_numpy(archive["validation_targets"].copy())
    if (
        features.shape != (len(rows), 4, 1154)
        or not torch.equal(cached_targets, targets)
        or not torch.isfinite(features).all()
    ):
        raise ValueError("frozen feature rows do not match the pinned validation corpus")

    checkpoint = run_dir / "matched_screen" / "checkpoint-best.safetensors"
    source_predictions = run_dir / "matched_screen" / "validation-predictions.jsonl"
    saved = [
        json.loads(line)
        for line in source_predictions.read_text(encoding="utf-8").splitlines()
    ]
    if len(saved) != len(rows):
        raise ValueError("saved prediction count differs from validation rows")

    model = MiniWoBOptionScorer().cpu()
    model.load_state_dict(load_file(checkpoint, device="cpu"), strict=True)
    model.eval()
    output_records = []
    max_probability_delta = 0.0
    argmax_changes = 0
    for row_index, (row, reference) in enumerate(zip(rows, saved, strict=True)):
        options = row["options"]
        option_ids = [option["option_id"] for option in options]
        if row["sample_id"] != reference["sample_id"]:
            raise ValueError("frozen corpus and saved predictions have different row order")
        result = model.decide_json(features[row_index : row_index + 1], option_ids)
        reference_probs = reference["probabilities_by_option_id"]
        actual_probs = {
            item["option_id"]: item["probability"] for item in result["options"]
        }
        deltas = [
            abs(actual_probs[option_id] - reference_probs[option_id])
            for option_id in option_ids
        ]
        max_probability_delta = max(max_probability_delta, *deltas)
        argmax_changes += int(result["selected_option_id"] != reference["selected_option_id"])
        if result["selected_option_id"] != reference["selected_option_id"]:
            raise RuntimeError("JSON selected option differs from the saved prediction")
        output_records.append({"sample_id": row["sample_id"], **result})

    if max_probability_delta > PROBABILITY_TOLERANCE:
        raise RuntimeError("JSON probabilities exceed the recorded CPU replay tolerance")

    prediction_path = output_dir / "validation-decisions.jsonl"
    prediction_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in output_records),
        encoding="utf-8",
    )
    report = {
        "status": "PASS_SAVED_PREDICTION_JSON_PARITY",
        "scope": "CPU inference/serialization parity only; no new model evaluation",
        "count": len(rows),
        "argmax_changes": argmax_changes,
        "max_probability_abs_difference": max_probability_delta,
        "probability_tolerance": PROBABILITY_TOLERANCE,
        "probability_parity": "within_tolerance",
        "temperature": 1.0,
        "hashes": {
            "effective_config": sha256_file(config_path),
            "validation_corpus": config["corpus"]["validation_jsonl_sha256"],
            "feature_cache": sha256_file(cache_path),
            "checkpoint": sha256_file(checkpoint),
            "reference_predictions": sha256_file(source_predictions),
            "replayed_predictions": sha256_file(prediction_path),
            "replay_script": sha256_file(Path(__file__).resolve()),
        },
    }
    (output_dir / "parity-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
