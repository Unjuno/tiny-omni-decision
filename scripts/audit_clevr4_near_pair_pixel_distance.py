"""Disambiguate low-pHash Clevr-4 candidate pairs with normalized pixel distances."""

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
from PIL import Image, ImageFilter

from scripts.audit_clevr4_content_overlap import _load_jsonl, phash_bytes, sha256_file


def normalized_rgb64(data: bytes) -> np.ndarray:
    with Image.open(BytesIO(data)) as image:
        resized = image.convert("RGB").resize((64, 64), Image.Resampling.LANCZOS)
        return np.asarray(resized, dtype=np.uint8)


def edge64(data: bytes) -> np.ndarray:
    with Image.open(BytesIO(data)) as image:
        edge = (
            image.convert("L")
            .filter(ImageFilter.FIND_EDGES)
            .resize((64, 64), Image.Resampling.LANCZOS)
        )
        return np.asarray(edge, dtype=np.uint8)


def _distance_vector(query: int, refs: np.ndarray) -> np.ndarray:
    xor = np.bitwise_xor(refs, np.uint64(query)).view(np.uint8).reshape(-1, 8)
    return np.unpackbits(xor, axis=1).sum(axis=1).astype(np.uint8, copy=False)


def run_audit(
    *,
    archive: Path,
    candidate_manifest: Path,
    old_train_images: Path,
    old_validation_images: Path,
    output: Path,
    near_distance: int = 4,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    candidates = _load_jsonl(candidate_manifest)
    old_paths = sorted(old_train_images.glob("*.png")) + sorted(old_validation_images.glob("*.png"))
    if not candidates or not old_paths:
        raise ValueError("candidate manifest and prior image directories must be nonempty")

    started = time.perf_counter()
    old_phashes: list[int] = []
    old_rgb: list[np.ndarray] = []
    old_edges: list[np.ndarray] = []
    for index, path in enumerate(old_paths, start=1):
        data = path.read_bytes()
        old_phashes.append(phash_bytes(data))
        old_rgb.append(normalized_rgb64(data))
        old_edges.append(edge64(data))
        if index % 2000 == 0:
            print(f"indexed prior image comparisons: {index}/{len(old_paths)}", flush=True)

    reference_hashes = np.asarray(old_phashes, dtype=np.uint64)
    candidate_near: list[dict[str, Any]] = []
    all_nearest: list[int] = []
    with zipfile.ZipFile(archive) as source:
        for index, row in enumerate(candidates, start=1):
            data = source.read(row["archive_member"])
            if hashlib.sha256(data).hexdigest() != row["media_sha256"]:
                raise ValueError(f"media SHA-256 mismatch for {row['id']}")
            distances = _distance_vector(phash_bytes(data), reference_hashes)
            all_nearest.append(int(distances.min()))
            candidates_within = np.flatnonzero(distances <= near_distance)
            if candidates_within.size:
                rgb = normalized_rgb64(data).astype(np.int16)
                edges = edge64(data).astype(np.int16)
                rgb_mae = [
                    float(np.abs(rgb - old_rgb[j].astype(np.int16)).mean())
                    for j in candidates_within
                ]
                edge_mae = [
                    float(np.abs(edges - old_edges[j].astype(np.int16)).mean())
                    for j in candidates_within
                ]
                best_index = int(candidates_within[int(np.argmin(rgb_mae))])
                best_metric = min(range(len(rgb_mae)), key=rgb_mae.__getitem__)
                candidate_near.append(
                    {
                        "candidate_id": row["id"],
                        "closest_reference_id_by_rgb64_mae": old_paths[best_index].stem,
                        "phash_distance": int(distances[best_index]),
                        "rgb64_mean_absolute_error_0_255": rgb_mae[best_metric],
                        "edge64_mean_absolute_error_0_255": edge_mae[best_metric],
                        "number_of_old_images_within_phash_threshold": int(candidates_within.size),
                    }
                )
            if index % 1000 == 0:
                print(f"compared candidate pairs: {index}/{len(candidates)}", flush=True)

    rgb_errors = [row["rgb64_mean_absolute_error_0_255"] for row in candidate_near]
    edge_errors = [row["edge64_mean_absolute_error_0_255"] for row in candidate_near]
    result = {
        "status": "COMPLETE_NEAR_PAIR_PIXEL_SCREEN_NOT_IDENTITY_PROOF",
        "candidate_count": len(candidates),
        "candidate_manifest_sha256": sha256_file(candidate_manifest),
        "old_image_count": len(old_paths),
        "near_phash_threshold": near_distance,
        "candidate_with_any_reference_within_threshold": len(candidate_near),
        "candidate_nearest_phash_min_median_p95": {
            "min": min(all_nearest),
            "median": statistics.median(all_nearest),
            "p95": float(np.percentile(all_nearest, 95)),
        },
        "best_rgb64_match_for_near_candidates": {
            "rgb_mae_min_median_p95": [
                min(rgb_errors),
                statistics.median(rgb_errors),
                float(np.percentile(rgb_errors, 95)),
            ],
            "edge_mae_min_median_p95": [
                min(edge_errors),
                statistics.median(edge_errors),
                float(np.percentile(edge_errors, 95)),
            ],
            "rgb_mae_le_1": sum(value <= 1 for value in rgb_errors),
            "rgb_mae_le_5": sum(value <= 5 for value in rgb_errors),
            "edge_mae_le_1": sum(value <= 1 for value in edge_errors),
            "edge_mae_le_5": sum(value <= 5 for value in edge_errors),
        },
        "near_pairs": candidate_near,
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
    parser.add_argument("--old-train-images", type=Path, required=True)
    parser.add_argument("--old-validation-images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--near-distance", type=int, default=4)
    args = parser.parse_args()
    print(
        json.dumps(
            run_audit(
                archive=args.archive,
                candidate_manifest=args.candidate_manifest,
                old_train_images=args.old_train_images,
                old_validation_images=args.old_validation_images,
                output=args.output,
                near_distance=args.near_distance,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
