"""Extract frozen 4x4 spatial-grid features for the fixed Clevr-4 inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tiny_omni_decision.dataset import sha256_file
from tiny_omni_decision.videomae_runtime import image_tensor
from tiny_omni_decision.videomae_spatial import FEATURE_WIDTH, GRID_SIZE, videomae_spatial_features

ARCHIVE_BYTES = 36_057_238_315
ENCODER_PARAMETERS = 22_153_286
ENCODER_SOURCE_SHA256 = "d132c97369f25207e380e78c59294fd00b93d2f6aa0fb5c4ded19eff26a5a3cb"
ENCODER_CHECKPOINT_SHA256 = "24fb71687fa3671b8387cadfbcbab0f72af695692e93cf1ecc82caa888626172"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def extract(
    *,
    archive: Path,
    manifest: Path,
    output: Path,
    batch_size: int = 8,
    max_images: int | None = None,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output {output}")
    if archive.stat().st_size != ARCHIVE_BYTES:
        raise ValueError("archive size differs from the pinned Clevr-4 preflight")
    if batch_size <= 0 or (max_images is not None and max_images <= 0):
        raise ValueError("batch_size and max_images must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("spatial frozen-feature extraction requires local CUDA")
    rows = _load_jsonl(manifest)
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("empty media manifest or duplicate IDs")
    if max_images is not None:
        rows = rows[:max_images]
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda")
    torch.manual_seed(17)
    torch.cuda.reset_peak_memory_stats(device)

    # Reuse the already pinned loader rather than introducing another model path.
    sys.path[:0] = ["src", "C:/CodexArtifacts/pretrained-reuse-r1/python-deps"]
    from scripts.extract_videomae_clevr4_frozen_features import load_encoder

    model = load_encoder().to(device).eval().requires_grad_(False)
    if sum(parameter.numel() for parameter in model.parameters()) != ENCODER_PARAMETERS:
        raise ValueError("loaded encoder parameter count differs from pinned model")
    started_at = datetime.now(UTC).isoformat()
    started = time.perf_counter()
    feature_path = output / "spatial-features.npy"
    matrix = np.lib.format.open_memmap(
        feature_path, mode="w+", dtype=np.float32, shape=(len(rows), FEATURE_WIDTH)
    )
    with zipfile.ZipFile(archive) as source:
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start : start + batch_size]
            tensors = []
            for row in batch_rows:
                image_bytes = source.read(row["archive_member"])
                if hashlib.sha256(image_bytes).hexdigest() != row["media_sha256"]:
                    raise ValueError(f"media SHA-256 mismatch for {row['id']}")
                tensors.append(image_tensor(image_bytes))
            inputs = torch.stack(tensors).to(device)
            with torch.inference_mode():
                encoded = videomae_spatial_features(model, inputs)
            if tuple(encoded.shape) != (len(batch_rows), FEATURE_WIDTH):
                raise ValueError(f"unexpected spatial feature shape {tuple(encoded.shape)}")
            if not torch.isfinite(encoded).all():
                raise ValueError("non-finite spatial encoder output")
            matrix[start : start + len(batch_rows)] = encoded.float().cpu().numpy()
            if (start + len(batch_rows)) % max(batch_size * 16, 128) == 0:
                print(f"encoded {start + len(batch_rows)}/{len(rows)} images", flush=True)
    matrix.flush()
    del matrix
    manifest_copy = output / "features-manifest.jsonl"
    manifest_copy.write_bytes(manifest.read_bytes())
    result = {
        "status": "COMPLETE_FROZEN_SPATIAL_GRID_FEATURE_EXTRACTION"
        if max_images is None
        else "SMOKE_FROZEN_SPATIAL_GRID_FEATURE_EXTRACTION",
        "started_at_utc": started_at,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "archive_bytes": archive.stat().st_size,
        "archive_sha512_recomputed": False,
        "input_manifest_sha256": sha256_file(manifest),
        "input_row_count": len(rows),
        "feature_shape": [len(rows), FEATURE_WIDTH],
        "feature_dtype": "float32",
        "feature_sha256": sha256_file(feature_path),
        "feature_manifest_sha256": sha256_file(manifest_copy),
        "encoder": "OpenGVLab VideoMAE V2 ViT-Small K710",
        "encoder_source_sha256": ENCODER_SOURCE_SHA256,
        "encoder_checkpoint_sha256": ENCODER_CHECKPOINT_SHA256,
        "encoder_parameters": ENCODER_PARAMETERS,
        "frozen": True,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "dtype": "float32",
        "batch_size": batch_size,
        "frames": 16,
        "temporal_reduction": "mean over temporal tubelets",
        "spatial_grid": [GRID_SIZE, GRID_SIZE],
        "spatial_reduction": "adaptive average pooling of final patch tokens",
        "normalization": "pretrained fc_norm applied independently to each pooled cell",
        "image_mode": "one static image repeated across frames; no temporal evidence",
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "wall_seconds": time.perf_counter() - started,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "reporting_partition_extracted": False,
        "reporting_partition_loaded": False,
        "sealed_audit_loaded": False,
    }
    (output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-images", type=int)
    args = parser.parse_args()
    extract(
        archive=args.archive,
        manifest=args.manifest,
        output=args.output,
        batch_size=args.batch_size,
        max_images=args.max_images,
    )


if __name__ == "__main__":
    main()
