"""Exercise one raw-video-to-Decision path across the separate runtime profiles."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tiny_omni_decision.video_runtime import (
    ClevrerBinaryVideoFeatureRuntime,
    sha256_file,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videoprism-python", type=Path, required=True)
    parser.add_argument("--videoprism-source", type=Path, required=True)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--assembly", type=Path, required=True)
    parser.add_argument("--validation-rows", type=Path, required=True)
    parser.add_argument("--expected-predictions", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()

    started = time.perf_counter()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite end-to-end output: {output}")
    assembly = args.assembly.resolve(strict=True)
    audio_component = assembly / "components/audio"
    video_component = assembly / "components/video"
    image_component = assembly / "components/image"
    rows = _read_jsonl(args.validation_rows.resolve(strict=True))
    references = _read_jsonl(args.expected_predictions.resolve(strict=True))
    if not rows or [str(row["id"]) for row in rows] != [
        str(row["id"]) for row in references
    ]:
        raise ValueError("validation rows and baseline predictions must have identical ID order")
    row = rows[0]
    reference = references[0]
    video = (args.sample_dir.resolve(strict=True) / str(row["media_path"])).resolve(strict=True)
    if sha256_file(video) != str(row["media_sha256"]):
        raise ValueError("validation video hash differs from the frozen sample row")
    with np.load(args.feature_cache.resolve(strict=True), allow_pickle=False) as cache:
        hashes = cache["media_sha256"].astype(str).tolist()
        feature_matrix = np.asarray(cache["features"], dtype=np.float32)
    feature_map = dict(zip(hashes, feature_matrix, strict=True))
    if str(row["media_sha256"]) not in feature_map:
        raise ValueError("reference pooled VideoPrism feature is missing")
    reference_feature = feature_map[str(row["media_sha256"])]

    output.mkdir(parents=True)
    worker_dir = output / "videoprism-worker"
    command = [
        str(args.videoprism_python.resolve(strict=True)),
        "-m",
        "scripts.encode_path_a_video_feature",
        "--video-path",
        str(video),
        "--expected-media-sha256",
        str(row["media_sha256"]),
        "--checkpoint",
        str(image_component / "videoprism-base-f16r288.npz"),
        "--source-root",
        str(args.videoprism_source.resolve(strict=True)),
        "--output-dir",
        str(worker_dir),
    ]
    worker = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    if worker.returncode:
        raise RuntimeError(
            "VideoPrism worker failed with exit "
            f"{worker.returncode}:\n{worker.stdout[-3000:]}\n{worker.stderr[-3000:]}"
        )
    worker_report = json.loads((worker_dir / "report.json").read_text(encoding="utf-8"))
    video_feature = np.load(worker_dir / "video-feature.npy", allow_pickle=False)
    feature_delta = float(np.max(np.abs(video_feature - reference_feature)))

    scorer_load_started = time.perf_counter()
    runtime = ClevrerBinaryVideoFeatureRuntime(
        audio_component / "models/text-minilm-en",
        video_component / "readout/best-readout.pt",
        device=args.device,
    )
    if args.device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    scorer_load_seconds = time.perf_counter() - scorer_load_started
    score_started = time.perf_counter()
    decision = runtime.decide(
        torch.from_numpy(video_feature),
        str(row.get("source_question", row["question"])),
        str(row["candidate_statement"]),
    )
    if args.device == "cuda":
        torch.cuda.synchronize()
    score_seconds = time.perf_counter() - score_started
    probabilities = decision.probabilities.tolist()
    logits = decision.option_logits.tolist()
    logit_delta = float(
        np.max(np.abs(np.asarray(logits, dtype=np.float32) - np.asarray(reference["logits"])))
    )
    probability_delta = float(
        np.max(
            np.abs(
                np.asarray(probabilities, dtype=np.float32)
                - np.asarray(reference["probabilities"], dtype=np.float32)
            )
        )
    )
    report = {
        "status": "complete_one_video_raw_to_decision_smoke",
        "created_utc": datetime.now(UTC).isoformat(),
        "claim_scope": (
            "single existing development-validation video; checks raw-video decode, "
            "VideoPrism feature extraction, and feature-input scorer wiring only"
        ),
        "sample_id": row["id"],
        "scene_index": row["scene_index"],
        "media_sha256": row["media_sha256"],
        "video_feature_max_abs_delta_vs_frozen_cache": feature_delta,
        "saved_prediction_index": int(reference["prediction_index"]),
        "runtime_prediction_index": decision.predicted_index,
        "prediction_mismatch": decision.predicted_index != int(reference["prediction_index"]),
        "max_logit_abs_delta_vs_saved_runtime": logit_delta,
        "max_probability_abs_delta_vs_saved_runtime": probability_delta,
        "videoprism_process_report": worker_report,
        "scorer_device": torch.cuda.get_device_name() if args.device == "cuda" else "cpu",
        "scorer_load_seconds": scorer_load_seconds,
        "scorer_inference_seconds": score_seconds,
        "scorer_peak_cuda_allocated_bytes": (
            torch.cuda.max_memory_allocated() if args.device == "cuda" else None
        ),
        "total_wall_seconds_including_worker_startup_model_load_and_scoring": (
            time.perf_counter() - started
        ),
        "sealed_audit_loaded": False,
        "quality_metrics_recomputed": False,
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
