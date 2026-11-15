"""Replay frozen MiniWoB screenshots through local DINOv2 plus Decision JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.train_miniwob_dino_path_a import read_corpus  # noqa: E402
from tiny_omni_decision.miniwob_dino_runtime import MiniWoBDinoRuntime, sha256_file  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--encoder-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve(strict=True)
    output_dir = args.output_dir.resolve()
    if run_dir == output_dir or run_dir in output_dir.parents:
        raise ValueError("runtime output must remain outside the frozen source run")
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "effective-config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _, _, splits = read_corpus(config)
    rows = splits["validation"]
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("limit must be positive")
        rows = rows[: args.limit]

    cache_path = run_dir / "frozen-features.npz"
    with np.load(cache_path, allow_pickle=False) as archive:
        cached_features = torch.from_numpy(archive["validation"].copy())
    reference_path = run_dir / "matched_screen" / "validation-predictions.jsonl"
    references = [
        json.loads(line)
        for line in reference_path.read_text(encoding="utf-8").splitlines()
    ]
    if (
        len(references) != len(splits["validation"])
        or cached_features.shape[0] != len(references)
    ):
        raise ValueError("frozen cache, validation corpus, and saved predictions differ in size")

    checkpoint = run_dir / "matched_screen" / "checkpoint-best.safetensors"
    runtime = MiniWoBDinoRuntime.from_local(args.encoder_dir, checkpoint)
    records = []
    max_feature_delta = 0.0
    max_probability_delta = 0.0
    selected_mismatches = 0
    batch_size = 8
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        batch_images = []
        batch_ids = []
        batch_coords = []
        for row in batch_rows:
            image_path = Path(config["corpus"]["path"]) / row["screenshot_path"]
            with Image.open(image_path) as image:
                batch_images.append(image.copy())
            batch_ids.append([option["option_id"] for option in row["options"]])
            batch_coords.append(
                [
                    (float(option["x_norm"]), float(option["y_norm"]))
                    for option in row["options"]
                ]
            )
        batch_features = runtime.encode_options_batch(batch_images, batch_coords)
        batch_decisions = [
            runtime.scorer.decide_json(batch_features[index : index + 1], option_ids)
            for index, option_ids in enumerate(batch_ids)
        ]
        for offset, (row, option_ids, features, decision) in enumerate(
            zip(batch_rows, batch_ids, batch_features, batch_decisions, strict=True)
        ):
            index = start + offset
            feature_delta = float((features - cached_features[index]).abs().max())
            max_feature_delta = max(max_feature_delta, feature_delta)
            reference = references[index]
            if row["sample_id"] != reference["sample_id"]:
                raise ValueError("saved predictions and frozen validation order disagree")
            actual = {
                option["option_id"]: option["probability"]
                for option in decision["options"]
            }
            reference_probs = reference["probabilities_by_option_id"]
            probability_delta = max(
                abs(actual[option_id] - reference_probs[option_id])
                for option_id in option_ids
            )
            max_probability_delta = max(max_probability_delta, probability_delta)
            mismatch = decision["selected_option_id"] != reference["selected_option_id"]
            selected_mismatches += int(mismatch)
            records.append(
                {
                    "sample_id": row["sample_id"],
                    **decision,
                    "feature_max_abs_difference_from_cache": feature_delta,
                    "probability_max_abs_difference_from_saved_batch": probability_delta,
                    "selected_id_matches_saved": not mismatch,
                }
            )

    prediction_path = output_dir / "image-runtime-decisions.jsonl"
    prediction_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    report = {
        "status": "PASS_SELECTED_ID_REPLAY" if selected_mismatches == 0 else "MISMATCH",
        "scope": "local CPU end-to-end screenshot-to-JSON replay; not a new quality evaluation",
        "count": len(records),
        "full_validation_replayed": len(records) == len(splits["validation"]),
        "selected_id_mismatches": selected_mismatches,
        "max_feature_abs_difference_from_cache": max_feature_delta,
        "max_probability_abs_difference_from_saved_batch": max_probability_delta,
        "hashes": {
            "effective_config": sha256_file(config_path),
            "feature_cache": sha256_file(cache_path),
            "readout_checkpoint": sha256_file(checkpoint),
            "validation_corpus": config["corpus"]["validation_jsonl_sha256"],
            "source_predictions": sha256_file(reference_path),
            "decision_output": sha256_file(prediction_path),
            "runtime_script": sha256_file(Path(__file__).resolve()),
        },
    }
    (output_dir / "image-runtime-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if selected_mismatches:
        raise RuntimeError("selected option IDs changed during local image-runtime replay")


if __name__ == "__main__":
    main()
