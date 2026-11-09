"""Bounded CUDA runtime smoke for pinned VideoMAEv2 weights; not a quality test."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np

from scripts.audit_path_a_videomaev2_roundtrip import (
    CHECKPOINT_SHA256,
    FRAME_COUNT,
    IMAGE_SIZE,
    MODEL_SOURCE_SHA256,
    _load_model,
    normalize_rgb_frames,
    static_image_clip,
)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def run_cuda_smoke(output: Path, *, repeats: int = 3) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    if repeats < 2:
        raise ValueError("repeats must be at least two for a repeatability check")

    torch, _module, model = _load_model()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; no CPU fallback is performed")
    device = torch.device("cuda:0")
    model = model.to(device).eval().requires_grad_(False)
    rng = np.random.default_rng(17)
    image = rng.integers(0, 256, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    video = rng.integers(
        0, 256, (FRAME_COUNT, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8
    )
    inputs = {
        "image_repeated_16_frames": normalize_rgb_frames(static_image_clip(image)),
        "synthetic_16_frame_video": normalize_rgb_frames(video),
    }

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    routes: dict[str, dict[str, Any]] = {}
    total_started = time.perf_counter()
    with torch.inference_mode():
        for route, values in inputs.items():
            batch = torch.from_numpy(values).to(device)
            model(batch)
            torch.cuda.synchronize(device)
            latencies: list[float] = []
            expected_logits = None
            expected_features = None
            for _ in range(repeats):
                started = time.perf_counter()
                logits = model(batch)
                features = model.forward_features(batch)
                torch.cuda.synchronize(device)
                latencies.append(time.perf_counter() - started)
                if expected_logits is None:
                    expected_logits = logits.detach().clone()
                    expected_features = features.detach().clone()
                elif not torch.equal(expected_logits, logits) or not torch.equal(
                    expected_features, features
                ):
                    raise ValueError(f"CUDA outputs were not repeatable for {route}")
            assert expected_logits is not None and expected_features is not None
            if not torch.isfinite(expected_logits).all() or not torch.isfinite(
                expected_features
            ).all():
                raise ValueError(f"non-finite output for {route}")
            routes[route] = {
                "input_sha256": _sha256(values.tobytes()),
                "logits_shape": list(expected_logits.shape),
                "features_shape": list(expected_features.shape),
                "finite": True,
                "bitwise_repeatable_within_run": True,
                "measured_forward_seconds": latencies,
                "median_forward_seconds": statistics.median(latencies),
            }
            del batch, logits, features, expected_logits, expected_features
    torch.cuda.synchronize(device)
    wall_seconds = time.perf_counter() - total_started
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "status": "PASS_CUDA_RUNTIME_SMOKE_NOT_QUALITY",
        "model_id": "OpenGVLab/VideoMAE2-vit-small-k710-distilled",
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "model_source_sha256": MODEL_SOURCE_SHA256,
        "device": torch.cuda.get_device_name(device),
        "device_total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "dtype": str(next(model.parameters()).dtype),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "frame_count": FRAME_COUNT,
        "repeats_per_route": repeats,
        "routes": routes,
        "wall_seconds": wall_seconds,
        "claims": {
            "training_performed": False,
            "dataset_loaded": False,
            "validation_scored": False,
            "decision_quality_measured": False,
            "sealed_audit_opened": False,
            "path_a_dispatcher_integrated": False,
        },
    }
    (output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    print(json.dumps(run_cuda_smoke(args.output, repeats=args.repeats), indent=2))


if __name__ == "__main__":
    main()
