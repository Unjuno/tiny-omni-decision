"""Extract frozen MOMENT per-channel/per-patch features for EgoViz sync windows."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

CHECKPOINT_SHA256 = "785e6c6f57ffa7cac7e2a1fff6369618d49f2f441563ddea76c866231e5aa877"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract(
    corpus_dir: Path,
    checkpoint_dir: Path,
    moment_source: Path,
    output: Path,
    batch_size: int = 16,
    device: str = "cuda",
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    checkpoint = checkpoint_dir / "model.safetensors"
    if sha256_file(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("MOMENT checkpoint hash mismatch")
    samples = [
        json.loads(line)
        for split in ("train", "development")
        for line in (corpus_dir / f"{split}-samples.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    candidate_indices = sorted(
        {index for row in samples for index in row["candidate_window_indices"]}
    )
    if candidate_indices != list(range(len(candidate_indices))):
        raise ValueError("candidate window indexes must cover a dense zero-based range")
    windows_path = corpus_dir / "imu-windows.npy"
    corpus = json.loads((corpus_dir / "window-corpus.json").read_text(encoding="utf-8"))
    if corpus["sha256"]["imu-windows.npy"] != sha256_file(windows_path):
        raise ValueError("IMU window tensor hash mismatch")
    windows = np.load(windows_path, allow_pickle=False)
    if windows.shape != (len(candidate_indices), 6, 128) or not np.isfinite(windows).all():
        raise ValueError(f"invalid IMU window tensor: {windows.shape}")

    import torch

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.startswith("cuda"):
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats()
    import sys

    sys.path.insert(0, str(moment_source))
    from momentfm import MOMENTPipeline

    model = MOMENTPipeline.from_pretrained(
        str(checkpoint_dir), model_kwargs={"task_name": "embedding"}, local_files_only=True
    )
    model.init()
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != 35_337_408:
        raise ValueError(f"unexpected loaded MOMENT parameter count: {parameter_count}")

    started = time.perf_counter()
    features: list[np.ndarray] = []
    for start in range(0, len(windows), batch_size):
        batch = torch.from_numpy(windows[start : start + batch_size]).to(
            device=device, dtype=torch.float32
        )
        mask = torch.ones((batch.shape[0], batch.shape[-1]), device=device, dtype=torch.bool)
        with torch.inference_mode():
            output_features = model(x_enc=batch, input_mask=mask, reduction="none").embeddings
        if output_features.shape[1:] != (6, 16, 512) or not torch.isfinite(output_features).all():
            raise ValueError(f"invalid MOMENT embedding shape/content: {output_features.shape}")
        features.append(output_features.cpu().numpy().astype(np.float32, copy=False))

    feature_array = np.concatenate(features, axis=0)
    output.mkdir(parents=True, exist_ok=False)
    cache_path = output / "sensor-features.npz"
    np.savez_compressed(
        cache_path,
        window_index=np.arange(len(feature_array), dtype=np.int32),
        features=feature_array,
    )
    report = {
        "status": "FROZEN_FEATURES_EXTRACTED_NO_TRAINING_OR_QUALITY_EVALUATION",
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "runner_sha256": sha256_file(Path(__file__)),
        "moment_source_commit": "38f7310ad594100747ca2a8357e9c7ca7d323e0e",
        "parameter_count": parameter_count,
        "feature_shape": list(feature_array.shape),
        "feature_semantics": "MOMENT embedding with per-channel and temporal patch axes retained",
        "source_window_count": len(windows),
        "batch_size": batch_size,
        "device": device,
        "torch_version": torch.__version__,
        "transformers_version": importlib.metadata.version("transformers"),
        "feature_file": cache_path.name,
        "feature_file_sha256": sha256_file(cache_path),
        "feature_file_bytes": cache_path.stat().st_size,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_bytes": (
            torch.cuda.max_memory_allocated() if device.startswith("cuda") else None
        ),
        "peak_reserved_bytes": (
            torch.cuda.max_memory_reserved() if device.startswith("cuda") else None
        ),
        "sealed_audit_accessed": False,
        "training": False,
    }
    (output / "feature-report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--moment-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    report = extract(
        args.corpus,
        args.checkpoint_dir,
        args.moment_source,
        args.output,
        args.batch_size,
        args.device,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
