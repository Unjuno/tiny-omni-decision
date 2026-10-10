"""Measure deterministic per-scene V-JEPA feature reuse on a pinned CLEVRER sample."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import av
import torch

from scripts.train_frozen_image_probe import _load_vjepa
from scripts.train_frozen_video_probe import (
    _decode_video,
    _load_rows,
    _validate_splits,
    _video_features,
)
from tiny_omni_decision.dataset import sha256_file


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--vjepa-source", type=Path, required=True)
    parser.add_argument("--vjepa-checkpoint", type=Path, required=True)
    parser.add_argument("--vjepa-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _uncached_rows(
    rows: list[dict[str, Any]], root: Path, encoder: Any, transform: Any, device: torch.device
) -> tuple[list[torch.Tensor], float]:
    features = []
    started = time.perf_counter()
    for row in rows:
        frames = _decode_video(root / row["media_path"])
        clip = transform(frames)[0].unsqueeze(0).to(device=device, dtype=torch.float32)
        with torch.inference_mode():
            features.append(encoder(clip).float().mean(dim=1).squeeze(0).cpu())
    torch.cuda.synchronize(device)
    return features, time.perf_counter() - started


def main() -> None:
    args = _args()
    run_started_at = datetime.now(UTC).isoformat()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    if args.vjepa_sha256 != "848a77c33cc9e6649ed2119c9bea1e2c569bcdab9539ff3e7c02ccc2959ddf4d":
        raise ValueError("unexpected V-JEPA checkpoint SHA-256")
    if not torch.cuda.is_available():
        raise RuntimeError("this V-JEPA cache diagnostic requires the existing local CUDA GPU")
    device = torch.device("cuda")
    torch.set_num_threads(4)
    torch.cuda.reset_peak_memory_stats(device)

    root = args.sample_dir.resolve()
    train = _load_rows(root / "train.jsonl")
    validation = _load_rows(root / "validation.jsonl")
    _validate_splits(root, train, validation)
    by_scene: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in validation:
        by_scene[int(row["scene_index"])].append(row)
    selected_scene, selected_rows = next(
        (scene, rows) for scene, rows in sorted(by_scene.items()) if len(rows) >= 2
    )
    if len({row["media_sha256"] for row in selected_rows}) != 1:
        raise ValueError("questions grouped under one scene have inconsistent video hashes")

    vjepa_args = SimpleNamespace(
        vjepa_source=args.vjepa_source,
        vjepa_checkpoint=args.vjepa_checkpoint,
        vjepa_sha256=args.vjepa_sha256,
    )
    encoder, transform, source_commit = _load_vjepa(vjepa_args, torch)
    encoder.eval().to(device)
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    model_allocated_bytes = torch.cuda.memory_allocated(device)

    torch.cuda.synchronize(device)
    cache_started = time.perf_counter()
    cached, _, unique_media_count, cache = _video_features(
        selected_rows, root, encoder, transform, device
    )
    torch.cuda.synchronize(device)
    cache_seconds = time.perf_counter() - cache_started
    uncached, uncached_seconds = _uncached_rows(
        selected_rows, root, encoder, transform, device
    )
    max_abs_difference = max(
        float((cached[index] - uncached[index]).abs().max().item())
        for index in range(len(selected_rows))
    )
    parity = all(
        torch.allclose(cached[index], uncached[index], rtol=1e-5, atol=1e-5)
        for index in range(len(selected_rows))
    )
    if not parity:
        raise ValueError(f"cached and uncached video features differ: max_abs={max_abs_difference}")

    cache_payload = {
        "scene_index": selected_scene,
        "question_ids": [row["id"] for row in selected_rows],
        "media_sha256": selected_rows[0]["media_sha256"],
        "cached_features": cached,
        "uncached_features": uncached,
    }
    feature_path = output / "parity-features.pt"
    torch.save(cache_payload, feature_path)
    report = {
        "schema_version": 1,
        "status": "complete_video_feature_cache_parity_diagnostic",
        "scope": "one pinned CLEVRER validation scene; feature parity and event-local reuse only",
        "started_utc": run_started_at,
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip(),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "sample_dir": str(root),
        "train_manifest_sha256": sha256_file(root / "train.jsonl"),
        "validation_manifest_sha256": sha256_file(root / "validation.jsonl"),
        "scene_index": selected_scene,
        "question_ids": [row["id"] for row in selected_rows],
        "question_count": len(selected_rows),
        "video_sha256": selected_rows[0]["media_sha256"],
        "split": "validation",
        "sealed_audit_loaded": False,
        "vjepa_source_commit": source_commit,
        "vjepa_checkpoint_sha256": args.vjepa_sha256,
        "frame_count": 8,
        "feature_shape": list(cached.shape),
        "feature_dtype": str(cached.dtype),
        "cache_unique_media": unique_media_count,
        "encoder_executions_with_cache": unique_media_count,
        "encoder_executions_without_cache": len(selected_rows),
        "encoder_executions_avoided": len(selected_rows) - unique_media_count,
        "cached_vs_uncached_allclose": parity,
        "max_abs_feature_difference": max_abs_difference,
        "cache_path_seconds": cache_seconds,
        "uncached_path_seconds": uncached_seconds,
        "measured_speedup_ratio_uncached_over_cached": uncached_seconds / cache_seconds,
        "unique_cached_feature_bytes": next(iter(cache.values())).nelement()
        * next(iter(cache.values())).element_size(),
        "question_expanded_feature_bytes": cached.nelement() * cached.element_size(),
        "cuda_model_allocated_bytes_before_inference": model_allocated_bytes,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "device": torch.cuda.get_device_name(device),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "transformers": importlib.metadata.version("transformers"),
            "av": av.__version__,
            "attention_backend": "PyTorch SDPA via pinned V-JEPA factory",
        },
        "feature_artifact": {
            "path": str(feature_path),
            "bytes": feature_path.stat().st_size,
            "sha256": sha256_file(feature_path),
        },
        "note": (
            "Single scene and two or more existing validation questions; cache correctness "
            "is not a "
            "product latency benchmark or evidence of temporal reasoning quality."
        ),
    }
    report["finished_utc"] = datetime.now(UTC).isoformat()
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
