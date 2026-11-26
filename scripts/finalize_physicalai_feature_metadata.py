"""Finalize provenance metadata for an already extracted validation feature cache."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

MODEL_REVISION = "f9e44c814b77203eaa57a6bdbbd535f21ede1415"


def finalize_metadata(
    artifact_dir: Path,
    *,
    device: str,
    batch_size: int,
    decode_seconds: float | None,
    inference_seconds: float | None,
    peak_cuda_bytes: int | None,
) -> dict[str, Any]:
    receipt_path = artifact_dir / "download-receipt.json"
    features_path = artifact_dir / "features.npz"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    features = np.load(features_path, allow_pickle=False)
    expected_ids = set(receipt["validation_episode_indices"])
    actual_ids = set(features["episode"].astype(int).tolist())
    if actual_ids != expected_ids:
        raise ValueError("feature episode IDs do not match the pinned validation receipt")
    if not np.all(features["split"] == "validation"):
        raise ValueError("feature archive includes a non-validation row")
    counts = {int(episode): int(np.sum(features["episode"] == episode)) for episode in actual_ids}
    if any(count != 4 for count in counts.values()):
        raise ValueError("every validation episode must have exactly four sampled queries")
    metadata = {
        "source_revision": receipt["revision"],
        "manifest_sha256": receipt["manifest_sha256"],
        "model_revision": MODEL_REVISION,
        "device": device,
        "batch_size": batch_size,
        "dtype": "float32",
        "validation_episode_count": len(actual_ids),
        "validation_query_count": int(len(features["episode"])),
        "frames_per_query": 8,
        "timestamp_frame_verified": True,
        "reserved_holdout_accessed": False,
        "decode_seconds": decode_seconds,
        "inference_seconds": inference_seconds,
        "peak_cuda_bytes": peak_cuda_bytes,
        "feature_file_sha256": hashlib.sha256(features_path.read_bytes()).hexdigest(),
    }
    (artifact_dir / "feature-extraction.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--decode-seconds", type=float)
    parser.add_argument("--inference-seconds", type=float)
    parser.add_argument("--peak-cuda-bytes", type=int)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    print(
        json.dumps(
            finalize_metadata(
                args.artifact_dir,
                device=args.device,
                batch_size=args.batch_size,
                decode_seconds=args.decode_seconds,
                inference_seconds=args.inference_seconds,
                peak_cuda_bytes=args.peak_cuda_bytes,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
