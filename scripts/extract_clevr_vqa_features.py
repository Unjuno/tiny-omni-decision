"""Extract frozen ViLT pooled image/question features for a fixed CLEVR manifest."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import ViltConfig, ViltForQuestionAnswering, ViltProcessor

from tiny_omni_decision.vilt_compat import remove_legacy_position_ids

EXPECTED_WEIGHTS_SHA256 = "4d5f3409947b0369487ece7c5868f0040ceb67d25735dbb4ac5e99e03bab3a19"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    weights = args.model / "pytorch_model.bin"
    if sha256(weights) != EXPECTED_WEIGHTS_SHA256:
        raise ValueError("ViLT checkpoint weight SHA-256 mismatch")
    rows = read_rows(args.manifest)
    if not rows or len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("manifest must be non-empty with unique sample IDs")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    script_started = datetime.now(UTC)
    wall_start = time.perf_counter()
    config = ViltConfig.from_pretrained(args.model, local_files_only=True)
    model = ViltForQuestionAnswering(config)
    state = torch.load(weights, weights_only=True, map_location="cpu")
    model.load_state_dict(
        remove_legacy_position_ids(state, config.max_position_embeddings), strict=True
    )
    del state
    model.eval().requires_grad_(False).to(args.device)
    processor = ViltProcessor.from_pretrained(args.model, local_files_only=True)
    lengths = processor.tokenizer(
        [str(row["question"]) for row in rows], truncation=False
    )["input_ids"]
    if max(map(len, lengths)) > config.max_position_embeddings:
        raise ValueError("a question exceeds the pinned ViLT position table")
    model_load_seconds = time.perf_counter() - wall_start

    image_hashes: list[str] = []
    features: list[np.ndarray] = []
    feature_start = time.perf_counter()
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        images = []
        try:
            for row in batch:
                split = str(row["split"])
                image_path = args.images_root / split / str(row["image_filename"])
                image_hashes.append(sha256(image_path))
                with Image.open(image_path) as source:
                    images.append(source.convert("RGB"))
            inputs = processor(
                images=images,
                text=[str(row["question"]) for row in batch],
                padding=True,
                truncation=False,
                return_tensors="pt",
            ).to(args.device)
            with torch.inference_mode():
                pooled = model.vilt(**inputs).pooler_output
            values = pooled.float().cpu().numpy()
            if values.shape != (len(batch), 768) or not np.isfinite(values).all():
                raise FloatingPointError("invalid frozen feature batch")
            features.append(values)
        finally:
            for image in images:
                image.close()
        if start % (args.batch_size * 100) == 0:
            print(f"features {min(start + len(batch), len(rows))}/{len(rows)}", flush=True)

    matrix = np.concatenate(features, axis=0)
    args.output.mkdir(parents=True)
    feature_path = args.output / "pooled-features.npy"
    np.save(feature_path, matrix, allow_pickle=False)
    if not np.array_equal(np.load(feature_path, allow_pickle=False), matrix):
        raise OSError("feature cache read-after-write mismatch")
    metadata = {
        "status": "FROZEN_FEATURES_READY",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "started_utc": script_started.isoformat(),
        "completed_utc": datetime.now(UTC).isoformat(),
        "device": args.device,
        "batch_size": args.batch_size,
        "model_load_seconds": model_load_seconds,
        "feature_extraction_seconds": time.perf_counter() - feature_start,
        "total_elapsed_seconds": time.perf_counter() - wall_start,
        "checkpoint_sha256": EXPECTED_WEIGHTS_SHA256,
        "model_file_sha256": {
            path.name: sha256(path)
            for path in sorted(args.model.iterdir())
            if path.is_file()
        },
        "manifest_sha256": sha256(args.manifest),
        "feature_sha256": sha256(feature_path),
        "feature_shape": list(matrix.shape),
        "feature_dtype": str(matrix.dtype),
        "image_sha256_in_manifest_order": image_hashes,
        "trainable_encoder_parameters": 0,
        "loaded_model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "numpy", "Pillow")
        },
        "quality_evaluation": False,
        "peak_cuda_memory_bytes": (
            torch.cuda.max_memory_allocated() if args.device == "cuda" else None
        ),
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
