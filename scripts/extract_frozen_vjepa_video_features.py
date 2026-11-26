"""Extract frozen V-JEPA video features without fitting a readout."""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from scripts.train_frozen_image_probe import _load_vjepa
from scripts.train_frozen_video_probe import (
    FRAME_COUNT,
    _decode_video,
    _load_rows,
    _validate_splits,
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


def _extract(
    rows: list[dict[str, Any]],
    root: Path,
    encoder: nn.Module,
    transform: Any,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], float]:
    features: dict[str, torch.Tensor] = {}
    started = time.perf_counter()
    for row in rows:
        media_hash = str(row["media_sha256"])
        if media_hash in features:
            continue
        frames = _decode_video(root / str(row["media_path"]))
        clip = transform(frames)[0].unsqueeze(0).to(device=device, dtype=torch.float32)
        if clip.shape[2] != FRAME_COUNT:
            raise ValueError("V-JEPA preprocessing changed the fixed eight-frame input")
        with torch.inference_mode():
            tokens = encoder(clip)
            pooled = tokens.float().mean(dim=1).squeeze(0)
        if pooled.shape != (768,) or not torch.isfinite(pooled).all():
            raise ValueError(f"V-JEPA produced invalid pooled feature for {row['id']}")
        features[media_hash] = pooled.cpu()
    return features, time.perf_counter() - started


def main() -> None:
    args = _args()
    sample_dir = args.sample_dir.resolve(strict=True)
    checkpoint = args.vjepa_checkpoint.resolve(strict=True)
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output}")
    if not torch.cuda.is_available():
        raise RuntimeError("the pinned V-JEPA video feature extractor requires local CUDA")

    train = _load_rows(sample_dir / "train.jsonl")
    validation = _load_rows(sample_dir / "validation.jsonl")
    _validate_splits(sample_dir, train, validation)
    started_at = datetime.now(UTC).isoformat()
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    torch.set_num_threads(4)
    encoder, transform, source_commit = _load_vjepa(args, torch)
    encoder.to(device)
    rows = train + validation
    features, extraction_seconds = _extract(rows, sample_dir, encoder, transform, device)
    expected = {str(row["media_sha256"]) for row in rows}
    if set(features) != expected:
        raise ValueError("extracted V-JEPA media identities differ from the sample manifests")

    output.mkdir(parents=True, exist_ok=False)
    feature_path = output / "video-features.npz"
    media_hashes = sorted(features)
    np.savez_compressed(
        feature_path,
        media_sha256=np.asarray(media_hashes),
        features=torch.stack([features[key] for key in media_hashes]).numpy(),
    )
    report = {
        "status": "complete_frozen_vjepa_video_feature_cache",
        "model": "facebook/vjepa2.1-vitb-dist-vitg-384",
        "upstream_source_commit": source_commit,
        "checkpoint_sha256": args.vjepa_sha256,
        "checkpoint_bytes": checkpoint.stat().st_size,
        "dataset_revision": "MIT-IBM/CLEVRER@98b842082ba4f7c18b6b9e3f39145871782a65ef",
        "sample_manifest_sha256": {
            name: sha256_file(sample_dir / filename)
            for name, filename in (("train", "train.jsonl"), ("validation", "validation.jsonl"))
        },
        "train_ids": [row["id"] for row in train],
        "validation_ids": [row["id"] for row in validation],
        "train_scene_ids": sorted({int(row["scene_index"]) for row in train}),
        "validation_scene_ids": sorted({int(row["scene_index"]) for row in validation}),
        "feature_count": len(media_hashes),
        "feature_dimension": 768,
        "frame_count": FRAME_COUNT,
        "preprocessing": (
            "PyAV decode; endpoint-inclusive uniformly sampled 8 frames; "
            "official V-JEPA 2.1 preprocessor, crop_size=384, pretrained=False"
        ),
        "feature_file": feature_path.name,
        "feature_file_sha256": sha256_file(feature_path),
        "feature_file_bytes": feature_path.stat().st_size,
        "extractor_sha256": sha256_file(Path(__file__).resolve()),
        "feature_extraction_seconds": extraction_seconds,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(UTC).isoformat(),
    }
    (output / "cache-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
