"""Exercise the new reader on one previously verified train-only feature pair."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from scripts.extract_perception_test_text_features import sha256_file
from tiny_omni_decision.perception_video_readout import PerceptionVideoDecisionReadout

VIDEO_FEATURE_SHA256 = "3fe0a89cbd43fa65bd5c978edaf73073ac695864c17a08beddea0c9eb3bb10c6"
TEXT_FEATURE_SHA256 = "1e22b98f793c901f64c5f0a3e9dae179458df83a6b0eb7095bc19b8fad7149a8"


def run_smoke(
    video_tokens_path: Path, text_features_path: Path, text_report_path: Path, output_dir: Path
):
    video_tokens_path = video_tokens_path.resolve(strict=True)
    text_features_path = text_features_path.resolve(strict=True)
    text_report_path = text_report_path.resolve(strict=True)
    if output_dir.drive.upper() != "D:" or output_dir.exists():
        raise ValueError("use a new, nonexisting output directory on D:")
    if sha256_file(video_tokens_path) != VIDEO_FEATURE_SHA256:
        raise ValueError("verified train-video smoke feature hash mismatch")
    if sha256_file(text_features_path) != TEXT_FEATURE_SHA256:
        raise ValueError("frozen train text-feature hash mismatch")
    text_report = json.loads(text_report_path.read_text(encoding="utf-8"))
    if text_report.get("labels_used") is not False or text_report.get("split") != "train":
        raise ValueError("smoke requires label-free train text features")
    with np.load(text_features_path, allow_pickle=False) as archive:
        sample_id = str(archive["sample_ids"][0])
        video_id = str(archive["video_ids"][0])
        queries = archive["features"][0:1]
    video = np.load(video_tokens_path, allow_pickle=False)
    if sample_id != "perception-test/train/video_10001/0" or video_id != "video_10001":
        raise ValueError("first text feature is not paired with the verified smoke video")
    if video.shape != (8, 14, 14, 384) or queries.shape != (1, 3, 384):
        raise ValueError("feature smoke inputs do not match the fixed reader contract")
    if not np.isfinite(video).all() or not np.isfinite(queries).all():
        raise FloatingPointError("smoke features must be finite")
    if not torch.cuda.is_available():
        raise RuntimeError("RTX 3080 CUDA device is required for this runtime smoke")

    device = torch.device("cuda")
    model = PerceptionVideoDecisionReadout().eval().to(device)
    video_tensor = torch.from_numpy(video.reshape(1, -1, 384).copy()).to(device)
    query_tensor = torch.from_numpy(queries.copy()).to(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with torch.inference_mode():
        logits = model(video_tensor, query_tensor)
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    if logits.shape != (1, 3) or not torch.isfinite(logits).all():
        raise FloatingPointError("reader failed to return finite three-choice logits")

    report = {
        "status": "PASS_RUNTIME_SMOKE_ONLY",
        "video_id": video_id,
        "sample_id": sample_id,
        "video_feature_sha256": VIDEO_FEATURE_SHA256,
        "text_feature_sha256": TEXT_FEATURE_SHA256,
        "video_feature_shape": list(video.shape),
        "candidate_query_shape": list(queries.shape),
        "output_shape": list(logits.shape),
        "finite_logits": True,
        "reader_parameter_count": model.parameter_count(),
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "forward_seconds": elapsed,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "labels_used": False,
        "training_performed": False,
        "quality_metrics_computed": False,
        "official_test_opened": False,
        "sealed_audit_opened": False,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "smoke-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-tokens", type=Path, required=True)
    parser.add_argument("--text-features", type=Path, required=True)
    parser.add_argument("--text-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run_smoke(
        video_tokens_path=args.video_tokens,
        text_features_path=args.text_features,
        text_report_path=args.text_report,
        output_dir=args.output_dir,
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
