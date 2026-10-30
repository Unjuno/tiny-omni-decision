"""Verify fresh-process reload parity for the saved EgoViz temporal readout."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts.train_egoviz_vs_sync_readout import (
    JointTemporalScorer,
    SensorOnlyScorer,
    _load_samples,
    _tensor_data,
    metrics,
    sha256_file,
)


def verify(
    corpus_dir: Path,
    video_cache: Path,
    sensor_cache: Path,
    result_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    result = json.loads((result_dir / "readout-results.json").read_text(encoding="utf-8"))
    predictions = [
        json.loads(line)
        for line in (result_dir / "development-predictions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    expected_hashes = result["input_sha256"]
    observed_hashes = {
        "train_samples": sha256_file(corpus_dir / "train-samples.jsonl"),
        "development_samples": sha256_file(corpus_dir / "development-samples.jsonl"),
        "video_features": sha256_file(video_cache),
        "sensor_features": sha256_file(sensor_cache),
    }
    if observed_hashes != expected_hashes:
        raise ValueError("feature or frozen split input hash mismatch")
    joint_path = result_dir / result["joint_temporal"]["state_file"]
    sensor_path = result_dir / result["sensor_only"]["state_file"]
    observed_state_hashes = {
        "joint_temporal": sha256_file(joint_path),
        "sensor_only": sha256_file(sensor_path),
    }
    if observed_state_hashes != {
        "joint_temporal": result["joint_temporal"]["state_sha256"],
        "sensor_only": result["sensor_only"]["state_sha256"],
    }:
        raise ValueError("saved readout checkpoint hash mismatch")

    rows, video_by_id, sensor_features = _load_samples(corpus_dir, video_cache, sensor_cache)
    data = _tensor_data(rows, video_by_id, sensor_features, "development")
    video, candidates, targets, groups, sample_ids, query_indices = data
    if len(predictions) != len(targets) or [row["sample_id"] for row in predictions] != sample_ids:
        raise ValueError("saved prediction IDs/order differ from development corpus")
    joint_model = JointTemporalScorer()
    sensor_model = SensorOnlyScorer()
    joint_model.load_state_dict(torch.load(joint_path, map_location="cpu", weights_only=True))
    sensor_model.load_state_dict(torch.load(sensor_path, map_location="cpu", weights_only=True))
    joint_model.eval()
    sensor_model.eval()

    mismatched_video = video.clone()
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        ordered = indices[np.argsort(query_indices[indices])]
        shift = max(1, len(ordered) // 2)
        mismatched_video[ordered] = video[np.roll(ordered, shift)]

    started = time.perf_counter()
    with torch.inference_mode():
        joint_logits = joint_model(video, candidates).numpy()
        sensor_logits = sensor_model(candidates).numpy()
        mismatch_logits = joint_model(mismatched_video, candidates).numpy()
    joint_probabilities = torch.softmax(torch.from_numpy(joint_logits), dim=1).numpy()
    mismatch_probabilities = torch.softmax(torch.from_numpy(mismatch_logits), dim=1).numpy()
    saved_joint = np.asarray([row["joint_probabilities"] for row in predictions])
    saved_mismatch = np.asarray(
        [row["mismatched_video_probabilities"] for row in predictions]
    )
    joint_delta = float(np.max(np.abs(joint_probabilities - saved_joint)))
    mismatch_delta = float(np.max(np.abs(mismatch_probabilities - saved_mismatch)))
    if joint_delta > 1e-6 or mismatch_delta > 1e-6:
        raise ValueError("fresh-process probabilities differ from saved predictions")
    saved_joint_predictions = [row["joint_prediction"] for row in predictions]
    if not np.array_equal(joint_logits.argmax(axis=1), saved_joint_predictions):
        raise ValueError("joint prediction IDs changed after reload")
    if not np.array_equal(
        sensor_logits.argmax(axis=1), [row["sensor_only_prediction"] for row in predictions]
    ):
        raise ValueError("sensor-only prediction IDs changed after reload")
    if not np.array_equal(
        mismatch_logits.argmax(axis=1), [row["mismatched_video_prediction"] for row in predictions]
    ):
        raise ValueError("mismatched-video prediction IDs changed after reload")

    output_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "PASS_FRESH_PROCESS_READOUT_RELOAD_PARITY",
        "sample_count": len(targets),
        "development_groups": int(len(np.unique(groups))),
        "input_sha256": observed_hashes,
        "checkpoint_sha256": observed_state_hashes,
        "saved_prediction_sha256": sha256_file(result_dir / "development-predictions.jsonl"),
        "joint_max_probability_abs_difference": joint_delta,
        "mismatched_video_max_probability_abs_difference": mismatch_delta,
        "joint_metrics": metrics(joint_logits, targets.numpy()),
        "sensor_only_metrics": metrics(sensor_logits, targets.numpy()),
        "mismatched_video_metrics": metrics(mismatch_logits, targets.numpy()),
        "verification_seconds": time.perf_counter() - started,
        "sealed_audit_accessed": False,
        "training": False,
    }
    (output_dir / "reload-verification.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--video-cache", type=Path, required=True)
    parser.add_argument("--sensor-cache", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            verify(
                args.corpus,
                args.video_cache,
                args.sensor_cache,
                args.result_dir,
                args.output_dir,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
