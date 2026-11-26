"""Read-only ZIP-to-image exposure screen for the Clevr-4 100k internal holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def phash_bytes(data: bytes) -> int:
    """Compute a deterministic 64-bit DCT perceptual hash for an image."""
    with Image.open(BytesIO(data)) as image:
        pixels = np.asarray(image.convert("L").resize((32, 32), Image.Resampling.LANCZOS))
    size = 32
    coords = np.arange(size, dtype=np.float64)
    frequencies = np.arange(size, dtype=np.float64)[:, None]
    dct = np.cos(np.pi * (2 * coords + 1) * frequencies / (2 * size))
    dct[0] *= 1.0 / np.sqrt(2.0)
    dct *= np.sqrt(2.0 / size)
    low = (dct @ pixels.astype(np.float64) @ dct.T)[:8, :8].reshape(-1)[1:]
    median = float(np.median(low))
    value = 0
    for bit in low > median:
        value = (value << 1) | int(bit)
    return value


def normalized_rgb64_sha256(data: bytes) -> str:
    """Hash canonical RGB pixels after a fixed 64x64 resize."""
    with Image.open(BytesIO(data)) as image:
        pixels = image.convert("RGB").resize((64, 64), Image.Resampling.LANCZOS)
        return hashlib.sha256(pixels.tobytes()).hexdigest()


def phash_distances(query: int, references: np.ndarray) -> np.ndarray:
    xor = np.bitwise_xor(references, np.uint64(query)).view(np.uint8).reshape(-1, 8)
    bit_counts = np.unpackbits(xor, axis=1).sum(axis=1)
    return bit_counts.astype(np.uint8, copy=False)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def run_audit(
    *,
    archive: Path,
    candidate_manifest: Path,
    previous_train: Path,
    previous_validation: Path,
    output: Path,
    expected_archive_bytes: int,
    expected_archive_sha512: str,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    if archive.stat().st_size != expected_archive_bytes:
        raise ValueError("archive size differs from its pinned preflight")
    candidates = _load_jsonl(candidate_manifest)
    if not candidates or len({row["id"] for row in candidates}) != len(candidates):
        raise ValueError("candidate media manifest is empty or has duplicate IDs")
    old_paths = sorted(previous_train.glob("*.png")) + sorted(previous_validation.glob("*.png"))
    if not old_paths:
        raise ValueError("no prior local Clevr-4 images found")

    started = time.perf_counter()
    old_hashes: set[str] = set()
    old_rgb64_hashes: set[str] = set()
    old_phashes: list[int] = []
    for index, path in enumerate(old_paths, start=1):
        data = path.read_bytes()
        old_hashes.add(hashlib.sha256(data).hexdigest())
        old_rgb64_hashes.add(normalized_rgb64_sha256(data))
        old_phashes.append(phash_bytes(data))
        if index % 2000 == 0:
            print(f"indexed prior images: {index}/{len(old_paths)}", flush=True)

    new_hashes: set[str] = set()
    new_rgb64_hashes: set[str] = set()
    new_phashes: list[int] = []
    with zipfile.ZipFile(archive) as source:
        for index, row in enumerate(candidates, start=1):
            data = source.read(row["archive_member"])
            digest = hashlib.sha256(data).hexdigest()
            if digest != row["media_sha256"]:
                raise ValueError(f"media SHA-256 mismatch for {row['id']}")
            new_hashes.add(digest)
            new_rgb64_hashes.add(normalized_rgb64_sha256(data))
            new_phashes.append(phash_bytes(data))
            if index % 1000 == 0:
                print(f"screened candidate images: {index}/{len(candidates)}", flush=True)

    prior_ids = {path.stem for path in old_paths}
    candidate_ids = {row["id"] for row in candidates}
    prior_hash_array = np.asarray(old_phashes, dtype=np.uint64)
    nearest: list[int] = []
    for value in new_phashes:
        nearest.append(int(phash_distances(value, prior_hash_array).min()))
    result = {
        "status": "COMPLETE_PHASH_SCREEN_NOT_PROOF_OF_ALL_CONTENT_IDENTITY",
        "scope": "all rows in the frozen 8,002-image internal validation media manifest",
        "archive": str(archive),
        "archive_bytes": archive.stat().st_size,
        "pinned_archive_sha512_from_preflight": expected_archive_sha512,
        "archive_sha512_recomputed": False,
        "candidate_manifest_sha256": sha256_file(candidate_manifest),
        "candidate_count": len(candidates),
        "candidate_id_order_sha256": hashlib.sha256(
            ("\n".join(row["id"] for row in candidates) + "\n").encode()
        ).hexdigest(),
        "previous_image_count": len(old_paths),
        "previous_image_manifest_sha256": {
            "train": sha256_file(previous_train.parent.parent / "train-image-manifest.jsonl"),
            "validation": sha256_file(
                previous_validation.parent.parent / "val-image-manifest.jsonl"
            ),
        },
        "candidate_old_id_overlap": len(candidate_ids & prior_ids),
        "candidate_old_media_sha256_overlap": len(new_hashes & old_hashes),
        "candidate_old_normalized_rgb64_pixel_hash_overlap": len(
            new_rgb64_hashes & old_rgb64_hashes
        ),
        "phash": {
            "algorithm": (
                "grayscale 32x32 Lanczos; orthonormal 2D DCT; low 8x8 "
                "excluding DC; median threshold"
            ),
            "bits": 63,
            "nearest_hamming_min": min(nearest),
            "nearest_hamming_median": statistics.median(nearest),
            "nearest_hamming_p95": float(np.percentile(nearest, 95)),
            "nearest_distance_le_2": sum(value <= 2 for value in nearest),
            "nearest_distance_le_4": sum(value <= 4 for value in nearest),
            "interpretation": (
                "heuristic similarity screen only; low distance requires "
                "inspection and is not identity proof"
            ),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "source_files_modified": False,
        "media_files_written": False,
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--previous-train-images", type=Path, required=True)
    parser.add_argument("--previous-validation-images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-archive-bytes", type=int, required=True)
    parser.add_argument("--expected-archive-sha512", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run_audit(
                archive=args.archive,
                candidate_manifest=args.candidate_manifest,
                previous_train=args.previous_train_images,
                previous_validation=args.previous_validation_images,
                output=args.output,
                expected_archive_bytes=args.expected_archive_bytes,
                expected_archive_sha512=args.expected_archive_sha512,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
