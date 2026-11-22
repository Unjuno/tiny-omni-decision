"""Fit one predeclared scalar temperature on the frozen calibration split only."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts.pretrained_reuse.evaluate_spoken_palette_dino_int8 import (
    DEFAULT_BASELINE,
    _load_source_colors,
    _make_encoder_config,
    quantize_dynamic_int8,
)
from scripts.pretrained_reuse.train_spoken_palette_decision import (
    arrays_for_rows,
    canonical_screen_rows,
    metrics,
    sha256_file,
    unique_screens,
    write_json,
)
from scripts.train_miniwob_dino_path_a import (
    load_encoder,
    model_option_positions,
    preprocess_screens,
)
from tiny_omni_decision.decision import FrozenFeatureCandidateScorer

DEFAULT_CORPUS = Path(
    r"C:\CodexArtifacts\pretrained-reuse-r2\spoken-palette-av-decision-v1-attempt-02"
)
DEFAULT_QUANTIZED = Path(
    r"C:\CodexArtifacts\pretrained-reuse-r2\spoken-palette-av-dino-dynamic-int8-attempt-02"
)
DEFAULT_OUTPUT = Path(
    r"C:\CodexArtifacts\pretrained-reuse-r2\spoken-palette-av-dino-int8-calibration-attempt-01"
)


def encode_quantized_features(
    model: torch.nn.Module, screens: list[dict[str, Any]], corpus_root: Path
) -> np.ndarray:
    rows = canonical_screen_rows(screens)
    feature_batches: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(rows), 8):
            batch = rows[start : start + 8]
            pixels = preprocess_screens(
                [corpus_root / row["screenshot_path"] for row in batch]
            )
            tokens = model(pixel_values=pixels).last_hidden_state
            patches = tokens[:, 1:].reshape(len(batch), 16, 16, 384)
            cls = tokens[:, 0]
            global_mean = patches.mean(dim=(1, 2))
            local_rows = []
            coordinates = []
            for row_index, row in enumerate(batch):
                local = []
                row_coordinates = []
                for x_norm, y_norm in model_option_positions(row):
                    x = min(int(x_norm * 16), 15)
                    y = min(int(y_norm * 16), 15)
                    x0, x1 = max(0, x - 1), min(16, x + 2)
                    y0, y1 = max(0, y - 1), min(16, y + 2)
                    local.append(patches[row_index, y0:y1, x0:x1].mean(dim=(0, 1)))
                    row_coordinates.append((x_norm, y_norm))
                local_rows.append(torch.stack(local))
                coordinates.append(row_coordinates)
            shared = torch.cat((cls, global_mean), dim=1)[:, None, :].expand(-1, 4, -1)
            feature_batches.append(
                torch.cat(
                    (
                        shared,
                        torch.stack(local_rows),
                        torch.tensor(coordinates, dtype=torch.float32),
                    ),
                    dim=-1,
                )
            )
    features = torch.cat(feature_batches).cpu().numpy().astype(np.float32, copy=False)
    if features.shape != (len(screens), 4, 1154) or not np.isfinite(features).all():
        raise ValueError("calibration screens produced malformed dynamic INT8 features")
    return features


def logits_for_rows(
    scorer: torch.nn.Module,
    arrays: dict[str, Any],
    audio_bank: np.ndarray,
    visual_bank: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    from scripts.pretrained_reuse.train_spoken_palette_decision import tensor_batch

    outputs = []
    scorer.eval()
    with torch.inference_mode():
        for start in range(0, len(arrays["targets"]), 512):
            indices = np.arange(start, min(start + 512, len(arrays["targets"])))
            features, _ = tensor_batch(indices, arrays, audio_bank, visual_bank, mean, std)
            outputs.append(scorer(features).float().cpu().numpy())
    logits = np.concatenate(outputs, axis=0)
    if logits.shape != (len(arrays["targets"]), 4) or not np.isfinite(logits).all():
        raise ValueError("calibration logits are malformed")
    return logits


def probabilities_at_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    scaled = logits.astype(np.float64) / temperature
    scaled -= scaled.max(axis=1, keepdims=True)
    exponentials = np.exp(scaled)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def run(
    corpus_root: Path,
    baseline_root: Path,
    quantized_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite calibration output: {output_root}")
    torch.set_num_threads(4)
    started = datetime.now(UTC)
    wall_start = time.perf_counter()

    baseline_result = json.loads((baseline_root / "result.json").read_text())
    effective = json.loads((baseline_root / "effective-config.json").read_text())
    quantized_result = json.loads((quantized_root / "result.json").read_text())
    corpus_manifest_path = corpus_root / "manifest.json"
    corpus_sha = sha256_file(corpus_manifest_path)
    corpus = json.loads(corpus_manifest_path.read_text(encoding="utf-8"))
    if corpus_sha != baseline_result["corpus_manifest_sha256"]:
        raise ValueError("baseline and calibration corpus do not match")
    if quantized_result["inputs"]["corpus_manifest_sha256"] != corpus_sha:
        raise ValueError("quantized encoder receipt is bound to another corpus")
    if corpus.get("evaluation_policy", {}).get("sealed_audit_opened") is not False:
        raise ValueError("source corpus audit state is not eligible")

    calibration_hash = corpus["split_row_sha256"]["calibration"]
    calibration_path = corpus_root / "calibration.jsonl"
    if sha256_file(calibration_path) != calibration_hash:
        raise ValueError("frozen calibration row hash mismatch")
    calibration_rows = [
        json.loads(line) for line in calibration_path.read_text(encoding="utf-8").splitlines()
    ]
    screens, screen_index = unique_screens(calibration_rows, corpus_root)

    state_path = quantized_root / quantized_result["artifacts"]["state_path"]
    expected_state_hash = quantized_result["quantization"]["quantized_state_sha256"]
    if sha256_file(state_path) != expected_state_hash:
        raise ValueError("dynamic INT8 state hash mismatch")
    encoder_config = _make_encoder_config(effective)
    base, _, base_hashes = load_encoder(encoder_config)
    if base_hashes != quantized_result["inputs"]["dino_base_files"]:
        raise ValueError("pinned DINOv2 base files differ from quantization receipt")
    quantized_model = quantize_dynamic_int8(base)
    del base
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    quantized_model.load_state_dict(state, strict=True)
    del state
    features = encode_quantized_features(quantized_model, screens, corpus_root)

    feature_path = output_root / "calibration-screen-features.npy"
    output_root.mkdir(parents=True, exist_ok=False)
    np.save(feature_path, features, allow_pickle=False)
    arrays = arrays_for_rows(
        calibration_rows, screen_index, _load_source_colors(effective)
    )
    scorer = FrozenFeatureCandidateScorer(1538, hidden_size=64).cpu().eval()
    checkpoint_path = baseline_root / "checkpoint-best.safetensors"
    expected_checkpoint_hash = baseline_result["selected_checkpoint"]["sha256"]
    if sha256_file(checkpoint_path) != expected_checkpoint_hash:
        raise ValueError("frozen float32 scorer checkpoint hash mismatch")
    from safetensors.torch import load_file

    scorer.load_state_dict(load_file(checkpoint_path, device="cpu"), strict=True)
    audio_root = Path(effective["source"]["audio_feature_path"])
    audio_bank = np.load(audio_root / "pooled.npy", mmap_mode="r", allow_pickle=False)
    normalizer = np.load(baseline_root / "train-normalizer.npz", allow_pickle=False)
    logits = logits_for_rows(
        scorer,
        arrays,
        audio_bank,
        features,
        normalizer["mean"],
        normalizer["std"],
    )

    temperatures = np.geomspace(0.25, 4.0, 81)
    calibration_scores = []
    for temperature in temperatures:
        probabilities = probabilities_at_temperature(logits, float(temperature))
        calibration_scores.append(
            metrics(probabilities, arrays["targets"], arrays["colors"])[
                "macro_color_nll"
            ]
        )
    best_index = int(np.argmin(calibration_scores))
    selected_temperature = float(temperatures[best_index])
    raw_temperature = float(baseline_result["selected_checkpoint"]["calibration_temperature"])
    selected_probabilities = probabilities_at_temperature(logits, selected_temperature)
    fixed_probabilities = probabilities_at_temperature(logits, raw_temperature)

    prediction_rows = []
    for row, probability in zip(calibration_rows, selected_probabilities, strict=True):
        option_ids = [option["option_id"] for option in row["options"]]
        prediction_rows.append(
            {
                "sample_id": row["sample_id"],
                "speaker_group": row["speaker_group"],
                "target_option_id": option_ids[row["target_index"]],
                "predicted_option_id": option_ids[int(probability.argmax())],
                "probabilities_by_option_id": {
                    key: float(value)
                    for key, value in zip(option_ids, probability, strict=True)
                },
            }
        )
    prediction_path = output_root / "calibration-predictions.jsonl"
    prediction_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in prediction_rows
        ),
        encoding="utf-8",
    )
    result = {
        "status": "PASS_CALIBRATION_ONLY_TEMPERATURE_FIT",
        "scope": "fixed scalar temperature for the dynamic INT8 DINOv2 AV pilot",
        "selection_or_recalibration": True,
        "calibration_only": True,
        "evaluation_rows_opened": False,
        "sealed_audit_opened": False,
        "started_utc": started.isoformat(),
        "ended_utc": datetime.now(UTC).isoformat(),
        "elapsed_seconds": time.perf_counter() - wall_start,
        "source_commit": os.environ.get("GIT_COMMIT", "unknown"),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": "cpu",
            "quantized_engine": torch.backends.quantized.engine,
            "threads": torch.get_num_threads(),
        },
        "inputs": {
            "corpus_manifest_sha256": corpus_sha,
            "calibration_rows_sha256": calibration_hash,
            "calibration_rows": len(calibration_rows),
            "calibration_speaker_groups": sorted(
                {row["speaker_group"] for row in calibration_rows}
            ),
            "unique_screens": len(screens),
            "int8_state_sha256": expected_state_hash,
            "float32_scorer_sha256": expected_checkpoint_hash,
            "prior_calibration_temperature": raw_temperature,
        },
        "selector": {
            "method": "minimum macro-color NLL on calibration split; earliest temperature on ties",
            "grid_min": float(temperatures[0]),
            "grid_max": float(temperatures[-1]),
            "grid_count": len(temperatures),
            "selected_index": best_index,
            "selected_temperature": selected_temperature,
        },
        "same_calibration_split_diagnostics_only": {
            "previous_temperature": metrics(
                fixed_probabilities, arrays["targets"], arrays["colors"]
            ),
            "selected_temperature": metrics(
                selected_probabilities, arrays["targets"], arrays["colors"]
            ),
        },
        "artifacts": {
            "features_sha256": sha256_file(feature_path),
            "features_bytes": feature_path.stat().st_size,
            "predictions_sha256": sha256_file(prediction_path),
            "predictions_path": prediction_path.name,
        },
        "limitations": [
            "Metrics are on the calibration examples used to choose temperature; they are "
            "in-sample diagnostics, not an independent quality estimate.",
            "No development evaluation or sealed audit rows were opened in this run.",
            "The task is constructed four-color speech-plus-screen action selection, not "
            "general multimodal Decision quality.",
            "Whisper embeddings and the Decision scorer remain float32; no complete package "
            "or device runtime was measured.",
        ],
    }
    write_json(output_root / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-root", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--quantized-root", type=Path, default=DEFAULT_QUANTIZED)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.corpus_root, args.baseline_root, args.quantized_root, args.output_root),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
