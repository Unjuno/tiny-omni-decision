"""Measure dynamic INT8 DINOv2 reuse on a frozen AV development pilot.

This is a fixed, non-selection quantization compatibility check. It does not
fit weights, tune calibration, or qualify a deployment package.
"""

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

from scripts.pretrained_reuse.train_spoken_palette_decision import (
    arrays_for_rows,
    canonical_screen_rows,
    infer,
    load_file,
    metrics,
    read_jsonl,
    sha256_file,
    unique_screens,
    write_json,
)
from scripts.train_miniwob_dino_path_a import load_encoder
from tiny_omni_decision.decision import FrozenFeatureCandidateScorer

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS = Path(
    r"C:\CodexArtifacts\pretrained-reuse-r2\spoken-palette-av-decision-v1-attempt-02"
)
DEFAULT_BASELINE = Path(
    r"C:\CodexArtifacts\pretrained-reuse-r2\spoken-palette-av-decision-v1-fit-attempt-01"
)
DEFAULT_OUTPUT = Path(
    r"C:\CodexArtifacts\pretrained-reuse-r2\spoken-palette-av-dino-dynamic-int8-attempt-01"
)


def quantize_dynamic_int8(module: torch.nn.Module) -> torch.nn.Module:
    """Return a CPU dynamic-qint8 copy; the source module remains untouched."""
    if any(parameter.device.type != "cpu" for parameter in module.parameters()):
        raise ValueError("dynamic INT8 pilot accepts CPU modules only")
    return torch.ao.quantization.quantize_dynamic(
        module, {torch.nn.Linear}, dtype=torch.qint8, inplace=False
    ).eval()


def _load_source_colors(config: dict[str, Any]) -> dict[str, str]:
    source_path = Path(config["source"]["audio_feature_path"]).parent
    rows_path = source_path / "lombard-grid-layout-02-media-gated" / "train.json"
    rows = json.loads(rows_path.read_text(encoding="utf-8"))
    return {row["source_id"]: row["targets"]["color"] for row in rows}


def _make_encoder_config(config: dict[str, Any]) -> dict[str, Any]:
    relative = Path(config["source"]["screen_encoder_config"])
    path = ROOT / relative
    if sha256_file(path) != config["source"]["screen_encoder_config_sha256"]:
        raise ValueError("pinned DINOv2 configuration hash mismatch")
    return json.loads(path.read_text(encoding="utf-8"))


def run(corpus_root: Path, baseline_root: Path, output_root: Path) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite quantization attempt: {output_root}")
    torch.set_num_threads(4)
    torch.manual_seed(17)
    np.random.seed(17)
    started = datetime.now(UTC)
    wall_start = time.perf_counter()

    effective = json.loads((baseline_root / "effective-config.json").read_text())
    baseline = json.loads((baseline_root / "result.json").read_text())
    corpus_manifest_path = corpus_root / "manifest.json"
    corpus_sha = sha256_file(corpus_manifest_path)
    corpus = json.loads(corpus_manifest_path.read_text(encoding="utf-8"))
    if corpus_sha != baseline["corpus_manifest_sha256"]:
        raise ValueError("baseline/corpus manifest identity mismatch")
    if corpus.get("evaluation_policy", {}).get("sealed_audit_opened") is not False:
        raise ValueError("source corpus audit state is not eligible for this pilot")

    expected_best = baseline["selected_checkpoint"]["sha256"]
    checkpoint_path = baseline_root / "checkpoint-best.safetensors"
    if sha256_file(checkpoint_path) != expected_best:
        raise ValueError("baseline selected checkpoint hash mismatch")
    baseline_feature_path = baseline_root / "evaluation-screen-features.npy"
    baseline_feature_sha = sha256_file(baseline_feature_path)
    if effective.get("config_sha256") != baseline.get("config_sha256"):
        raise ValueError("baseline config identity mismatch")

    row_hash = corpus["split_row_sha256"]["evaluation"]
    evaluation_rows = read_jsonl(corpus_root / "evaluation.jsonl", row_hash)
    screens, screen_index = unique_screens(evaluation_rows, corpus_root)
    screen_rows = canonical_screen_rows(screens)
    encoder_config = _make_encoder_config(effective)
    source_model, model_dir, model_hashes = load_encoder(encoder_config)
    quantize_start = time.perf_counter()
    quantized = quantize_dynamic_int8(source_model)
    quantize_seconds = time.perf_counter() - quantize_start
    del source_model

    output_root.mkdir(parents=True, exist_ok=False)
    quantized_state_path = output_root / "dinov2-small-dynamic-int8-state.pt"
    torch.save(quantized.state_dict(), quantized_state_path)
    # The payload was just created by this process; no untrusted pickle is loaded.
    state = torch.load(quantized_state_path, map_location="cpu", weights_only=False)
    source_model, _, _ = load_encoder(encoder_config)
    reloaded = quantize_dynamic_int8(source_model)
    reloaded.load_state_dict(state, strict=True)
    del state, source_model

    feature_start = time.perf_counter()
    feature_chunks: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(screens), 8):
            from scripts.train_miniwob_dino_path_a import preprocess_screens

            batch = screen_rows[start : start + 8]
            paths = [corpus_root / row["screenshot_path"] for row in batch]
            pixels = preprocess_screens(paths)
            tokens = reloaded(pixel_values=pixels).last_hidden_state
            cls = tokens[:, 0]
            patches = tokens[:, 1:].reshape(len(batch), 16, 16, 384)
            global_mean = patches.mean(dim=(1, 2))
            local_rows = []
            coords = []
            from scripts.train_miniwob_dino_path_a import model_option_positions

            for row_index, row in enumerate(batch):
                local = []
                current_coords = []
                for x_norm, y_norm in model_option_positions(row):
                    column = min(int(x_norm * 16), 15)
                    row_y = min(int(y_norm * 16), 15)
                    x0, x1 = max(0, column - 1), min(16, column + 2)
                    y0, y1 = max(0, row_y - 1), min(16, row_y + 2)
                    local.append(patches[row_index, y0:y1, x0:x1].mean(dim=(0, 1)))
                    current_coords.append((x_norm, y_norm))
                local_rows.append(torch.stack(local))
                coords.append(current_coords)
            shared = torch.cat((cls, global_mean), dim=1)[:, None, :].expand(-1, 4, -1)
            feature_chunks.append(
                torch.cat(
                    (
                        shared,
                        torch.stack(local_rows),
                        torch.tensor(coords, dtype=torch.float32),
                    ),
                    dim=-1,
                )
            )
    feature_seconds = time.perf_counter() - feature_start
    quantized_features = torch.cat(feature_chunks).cpu().numpy().astype(np.float32)
    if quantized_features.shape != (len(screens), 4, 1154) or not np.isfinite(
        quantized_features
    ).all():
        raise ValueError("dynamic INT8 DINOv2 produced invalid option features")
    quantized_features_path = output_root / "evaluation-screen-features.npy"
    np.save(quantized_features_path, quantized_features, allow_pickle=False)

    baseline_features = np.load(baseline_feature_path, allow_pickle=False)
    if baseline_features.shape != quantized_features.shape:
        raise ValueError("baseline and dynamic INT8 option feature layouts differ")
    feature_delta = np.abs(quantized_features - baseline_features)

    scorer = FrozenFeatureCandidateScorer(1538, hidden_size=64).cpu().eval()
    scorer.load_state_dict(load_file(checkpoint_path, device="cpu"), strict=True)
    audio_root = Path(effective["source"]["audio_feature_path"])
    audio_bank = np.load(audio_root / "pooled.npy", mmap_mode="r", allow_pickle=False)
    normalizer = np.load(baseline_root / "train-normalizer.npz", allow_pickle=False)
    colors = _load_source_colors(effective)
    arrays = arrays_for_rows(evaluation_rows, screen_index, colors)
    temperature = float(baseline["selected_checkpoint"]["calibration_temperature"])
    probabilities = infer(
        scorer,
        arrays,
        audio_bank,
        quantized_features,
        normalizer["mean"],
        normalizer["std"],
        temperature=temperature,
    )
    quantized_metrics = metrics(probabilities, arrays["targets"], arrays["colors"])

    baseline_prediction_path = baseline_root / "evaluation-predictions.jsonl"
    if sha256_file(baseline_prediction_path) != baseline["evaluation_prediction_sha256"]:
        raise ValueError("baseline prediction file hash mismatch")
    old_predictions = [
        json.loads(line)
        for line in baseline_prediction_path.read_text(encoding="utf-8").splitlines()
    ]
    baseline_probability_matrix = np.asarray(
        [
            [row["probabilities_by_option_id"][option_id] for option_id in row["option_ids"]]
            for row in old_predictions
        ],
        dtype=np.float64,
    )
    if len(old_predictions) != len(probabilities) or any(
        row["sample_id"] != source["sample_id"]
        for row, source in zip(old_predictions, evaluation_rows, strict=True)
    ):
        raise ValueError("baseline prediction IDs/order differ from frozen evaluation rows")
    probability_delta = np.abs(probabilities - baseline_probability_matrix)
    prediction_rows = []
    for row, probability in zip(evaluation_rows, probabilities, strict=True):
        option_ids = [candidate["option_id"] for candidate in row["options"]]
        prediction_rows.append(
            {
                "sample_id": row["sample_id"],
                "target_option_id": option_ids[row["target_index"]],
                "predicted_option_id": option_ids[int(probability.argmax())],
                "probabilities_by_option_id": {
                    option_id: float(value)
                    for option_id, value in zip(option_ids, probability, strict=True)
                },
            }
        )
    predictions_path = output_root / "evaluation-predictions.jsonl"
    predictions_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in prediction_rows
        ),
        encoding="utf-8",
    )

    result = {
        "status": "PASS_FIXED_DEVELOPMENT_QUANTIZATION_COMPATIBILITY_CHECK",
        "scope": "dynamic qint8 DINOv2-small plus frozen original audio, readout and temperature",
        "selection_or_recalibration": False,
        "sealed_audit_opened": False,
        "started_utc": started.isoformat(),
        "ended_utc": datetime.now(UTC).isoformat(),
        "elapsed_seconds": time.perf_counter() - wall_start,
        "source_commit": os.environ.get("GIT_COMMIT", "unknown"),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "quantized_engine": torch.backends.quantized.engine,
            "device": "cpu",
            "threads": torch.get_num_threads(),
        },
        "inputs": {
            "corpus_manifest_sha256": corpus_sha,
            "evaluation_rows_sha256": row_hash,
            "baseline_checkpoint_sha256": expected_best,
            "baseline_eval_feature_sha256": baseline_feature_sha,
            "dino_config_sha256": effective["source"]["screen_encoder_config_sha256"],
            "dino_revision": effective["source"]["screen_encoder_revision"],
            "dino_base_files": model_hashes,
            "dino_model_dir": str(model_dir),
            "temperature": temperature,
            "evaluation_examples": len(evaluation_rows),
            "unique_screens": len(screens),
        },
        "quantization": {
            "method": "PyTorch eager dynamic quantization of every nn.Linear to qint8",
            "encoder_quantize_seconds": quantize_seconds,
            "encoder_inference_seconds": feature_seconds,
            "quantized_state_bytes": quantized_state_path.stat().st_size,
            "quantized_state_sha256": sha256_file(quantized_state_path),
            "feature_file_bytes": quantized_features_path.stat().st_size,
            "feature_file_sha256": sha256_file(quantized_features_path),
            "max_abs_feature_delta": float(feature_delta.max()),
            "mean_abs_feature_delta": float(feature_delta.mean()),
            "max_abs_probability_delta": float(probability_delta.max()),
            "mean_abs_probability_delta": float(probability_delta.mean()),
            "argmax_flips": int(
                np.count_nonzero(probabilities.argmax(1) != baseline_probability_matrix.argmax(1))
            ),
            "state_reload": (
                "strict reload into a freshly reconstructed dynamic-qint8 architecture passed"
            ),
        },
        "baseline_float32_metrics": baseline["evaluation"]["matched"],
        "dynamic_int8_metrics_same_exposed_development_set": quantized_metrics,
        "artifacts": {
            "state_path": quantized_state_path.name,
            "feature_path": quantized_features_path.name,
            "prediction_path": predictions_path.name,
        },
        "limitations": [
            "The exposed development set is reused only for fixed quantization compatibility; "
            "it is not blind and did not select or calibrate anything.",
            "Whisper and the 98,561-parameter scorer remain float32; this is not a complete "
            "quantized model or package.",
            "The PyTorch state payload needs the pinned DINO architecture and quantized CPU "
            "runtime; dependency bytes and end-user runtime are excluded from its size.",
            "This synthetic four-color task does not establish natural image, video, audio, "
            "or product Decision quality.",
            "No training, GPU, cloud, or sealed-audit access occurred.",
        ],
    }
    write_json(output_root / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-root", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.corpus_root, args.baseline_root, args.output_root), indent=2))


if __name__ == "__main__":
    main()
