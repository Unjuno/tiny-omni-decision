"""Recheck joint STARSS23 input controls using the frozen selected checkpoint."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch

from scripts.extract_frozen_videoprism_features import sha256_file
from scripts.extract_starss23_corpus_features import read_manifest
from scripts.fetch_starss23_frozen_media import write_json
from scripts.train_starss23_spatial_decision import (
    EVENTS,
    HIDDEN_SIZE,
    SECTORS,
    _feature_matrix,
    _metric_summary,
    _predict,
    load_queries,
    same_room_video_permutation,
    zero_joint_modality_controls,
)
from tiny_omni_decision.decision import FrozenFeatureCandidateScorer


def audit(args: argparse.Namespace) -> dict:
    manifest_rows = read_manifest(args.manifest_dir)
    manifest_sha = sha256_file(args.manifest_dir / "result.json")
    audio, audio_report = _feature_matrix(
        args.audio_features, "audio", manifest_sha, (2720, 4, 384), manifest_rows
    )
    video, video_report = _feature_matrix(
        args.video_features, "video", manifest_sha, (2720, 8, 768), manifest_rows
    )
    splits, corpus_report = load_queries(args.corpus_dir, manifest_rows, manifest_sha)
    run_result = json.loads((args.run_dir / "result.json").read_text(encoding="utf-8"))
    if run_result.get("status") != "PASS":
        raise ValueError("selected readout run is incomplete")
    joint_arm = run_result["arms"]["audio_video_joint"]
    checkpoint = args.run_dir / "audio_video_joint" / "best.pt"
    if sha256_file(checkpoint) != joint_arm["best_sha256"]:
        raise ValueError("selected joint checkpoint hash mismatch")
    normalization_path = args.run_dir / "audio_video_joint" / "normalization.npz"
    if sha256_file(normalization_path) != joint_arm["normalization_sha256"]:
        raise ValueError("joint normalization hash mismatch")

    joint_raw = np.concatenate((audio, video), axis=1)
    with np.load(normalization_path, allow_pickle=False) as normalization:
        mean = normalization["mean"].astype(np.float32)
        std = normalization["std"].astype(np.float32)
    if mean.shape != (joint_raw.shape[1],) or std.shape != mean.shape:
        raise ValueError("joint normalization shape mismatch")
    joint = ((joint_raw - mean) / std).astype(np.float32)
    if not np.isfinite(joint).all():
        raise ValueError("nonfinite joint features")

    validation = splits["validation"]
    unique_windows = np.unique(validation["window"])
    recording_by_window = [
        validation["recording"][np.flatnonzero(validation["window"] == window)[0]]
        for window in unique_windows
    ]
    room_by_window = [
        validation["room"][np.flatnonzero(validation["window"] == window)[0]]
        for window in unique_windows
    ]
    replacements = same_room_video_permutation(
        unique_windows, recording_by_window, room_by_window
    )
    replacement_by_window = {
        int(window): int(replacements[index]) for index, window in enumerate(unique_windows)
    }
    permuted_validation = dict(validation)
    permuted_validation["window"] = np.asarray(
        [replacement_by_window[int(window)] for window in validation["window"]], dtype=np.int64
    )
    controls = zero_joint_modality_controls(joint, audio.shape[1])
    matrices = {
        "intact": (joint, validation),
        "video_zeroed_at_train_mean": (controls["video_zeroed"], validation),
        "audio_zeroed_at_train_mean": (controls["audio_zeroed"], validation),
        "video_from_different_recording_same_room": (joint, permuted_validation),
    }

    device = torch.device(args.device)
    model = FrozenFeatureCandidateScorer(
        joint.shape[1] + len(EVENTS) + len(SECTORS) + 2, HIDDEN_SIZE
    ).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    temperature = float(joint_arm["calibration"]["temperature"])
    yes_bias = float(joint_arm["calibration"]["yes_bias"])
    results = {}
    prediction_arrays = {
        "sample_ids": validation["ids"],
        "recording": validation["recording"],
        "room": validation["room"],
        "targets": validation["target"],
        "event_ids": validation["event"],
        "sector_ids": validation["sector"],
    }
    for name, (matrix, rows) in matrices.items():
        logits = _predict(model, matrix, rows, device)
        raw_probabilities = logits.softmax(-1)[:, 1].numpy()
        calibrated_logits = logits.clone()
        calibrated_logits[:, 1] += yes_bias * temperature
        calibrated_probabilities = (calibrated_logits / temperature).softmax(-1)[:, 1].numpy()
        results[name] = {
            "raw": _metric_summary(
                rows["target"], raw_probabilities, rows["event"], rows["sector"], rows["room"]
            ),
            "calibrated": _metric_summary(
                rows["target"],
                calibrated_probabilities,
                rows["event"],
                rows["sector"],
                rows["room"],
            ),
        }
        prediction_arrays[f"{name}_raw_yes_probability"] = raw_probabilities
        prediction_arrays[f"{name}_calibrated_yes_probability"] = calibrated_probabilities

    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    predictions_path = args.output_dir / "predictions.npz"
    np.savez_compressed(predictions_path, **prediction_arrays)
    report = {
        "status": "PASS",
        "scope": (
            "fixed selected audio_video_joint checkpoint; post-hoc input-control audit; "
            "no training"
        ),
        "validation_is_blind": False,
        "sample_count": len(validation["target"]),
        "unique_recordings": len(set(validation["recording"].tolist())),
        "unique_rooms": sorted(set(validation["room"].tolist())),
        "selected_checkpoint_sha256": joint_arm["best_sha256"],
        "normalization_sha256": joint_arm["normalization_sha256"],
        "manifest_sha256": manifest_sha,
        "corpus_sha256": sha256_file(args.corpus_dir / "result.json"),
        "audio_feature_sha256": audio_report["feature_sha256"],
        "video_feature_sha256": video_report["feature_sha256"],
        "video_permutation": "seed 17; different recording within same room; label order retained",
        "zero_controls": (
            "set selected standardized joint modality coordinates to zero (train mean)"
        ),
        "calibration": joint_arm["calibration"],
        "controls": results,
        "predictions_sha256": sha256_file(predictions_path),
        "device": str(device),
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
        },
    }
    write_json(args.output_dir / "result.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--audio-features", type=Path, required=True)
    parser.add_argument("--video-features", type=Path, required=True)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    report = audit(args)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
