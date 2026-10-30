"""Compare saved forward/reversed video features on identical validation assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _summary(values: np.ndarray) -> dict[str, float]:
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("summary values must be a non-empty finite vector")
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p05": float(np.quantile(values, 0.05)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def compare_features(
    forward: np.ndarray,
    reverse: np.ndarray,
    *,
    media_hashes: list[str],
) -> dict[str, Any]:
    if forward.ndim != 2 or reverse.shape != forward.shape:
        raise ValueError("forward and reverse features must have the same [scene, dim] shape")
    if len(media_hashes) != forward.shape[0] or len(set(media_hashes)) != len(media_hashes):
        raise ValueError("media hashes must be unique and match the feature row count")
    if not np.isfinite(forward).all() or not np.isfinite(reverse).all():
        raise ValueError("feature arrays must be finite")
    forward_norm = np.linalg.norm(forward, axis=1)
    reverse_norm = np.linalg.norm(reverse, axis=1)
    if np.any(forward_norm <= 0) or np.any(reverse_norm <= 0):
        raise ValueError("feature vectors must have non-zero norm")

    cosine = np.sum(forward * reverse, axis=1) / (forward_norm * reverse_norm)
    relative_l2 = np.linalg.norm(forward - reverse, axis=1) / forward_norm
    return {
        "paired_assets": len(media_hashes),
        "feature_dimension": int(forward.shape[1]),
        "bitwise_equal": bool(np.array_equal(forward, reverse)),
        "cosine_similarity": _summary(cosine),
        "relative_l2_difference": _summary(relative_l2),
    }


def _load_feature_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"media_sha256", "features"}:
            raise ValueError(f"unexpected feature cache fields: {path}")
        hashes = [str(value) for value in archive["media_sha256"].tolist()]
        features = np.asarray(archive["features"], dtype=np.float64)
    if features.ndim != 2 or features.shape[0] != len(hashes):
        raise ValueError(f"feature cache row count mismatch: {path}")
    if len(set(hashes)) != len(hashes):
        raise ValueError(f"duplicate media identity in feature cache: {path}")
    return dict(zip(hashes, features, strict=True))


def _load_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "complete_videoprism_frozen_feature_cache":
        raise ValueError(f"incomplete or unexpected cache report: {path}")
    return report


def analyze(
    *,
    forward_cache: Path,
    forward_report: Path,
    reverse_cache: Path,
    reverse_report: Path,
    validation_manifest: Path,
) -> dict[str, Any]:
    forward_meta = _load_report(forward_report)
    reverse_meta = _load_report(reverse_report)
    invariant_fields = (
        "model_id",
        "model_revision",
        "upstream_source_commit",
        "checkpoint_sha256",
        "dataset_revision",
        "frame_count",
        "frame_size",
        "feature_dimension",
        "validation_ids",
        "validation_media_hashes",
        "sample_manifest_sha256",
        "preprocessing",
        "pooling",
    )
    for field in invariant_fields:
        if forward_meta.get(field) != reverse_meta.get(field):
            raise ValueError(f"forward/reverse cache metadata differs: {field}")
    if forward_meta.get("temporal_order", "forward") != "forward":
        raise ValueError("forward cache is not marked as forward temporal order")
    if reverse_meta.get("temporal_order") != "reverse":
        raise ValueError("reverse cache is not marked as reversed temporal order")
    if _sha256(forward_cache) != forward_meta.get("feature_cache_sha256"):
        raise ValueError("forward cache hash does not match its report")
    if _sha256(reverse_cache) != reverse_meta.get("feature_cache_sha256"):
        raise ValueError("reverse cache hash does not match its report")

    rows = [
        json.loads(line)
        for line in validation_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    manifest_hashes = {str(row["media_sha256"]) for row in rows}
    if manifest_hashes != set(forward_meta["validation_media_hashes"]):
        raise ValueError("validation manifest assets differ from frozen cache report")
    forward_by_hash = _load_feature_cache(forward_cache)
    reverse_by_hash = _load_feature_cache(reverse_cache)
    forward_hashes = set(forward_by_hash)
    reverse_hashes = set(reverse_by_hash)
    if not manifest_hashes <= forward_hashes or not manifest_hashes <= reverse_hashes:
        raise ValueError("a validation asset is missing from one feature cache")
    ordered_hashes = sorted(manifest_hashes)
    forward = np.stack([forward_by_hash[key] for key in ordered_hashes])
    reverse = np.stack([reverse_by_hash[key] for key in ordered_hashes])
    if forward.shape[1] != int(forward_meta["feature_dimension"]):
        raise ValueError("feature width differs from frozen cache metadata")

    result = compare_features(forward, reverse, media_hashes=ordered_hashes)
    project_root = Path(__file__).resolve().parents[1]
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        cwd=project_root,
    ).stdout.strip()
    return {
        "schema_version": 1,
        "status": "complete_saved_feature_order_sensitivity",
        "scope": (
            "descriptive comparison of already-saved pooled validation features; "
            "no model fitting, checkpoint selection, or new quality evaluation"
        ),
        "source_commit": source_commit,
        "analysis_script_sha256": _sha256(Path(__file__).resolve()),
        "model": {
            key: forward_meta[key]
            for key in (
                "model_id",
                "model_revision",
                "upstream_source_commit",
                "checkpoint_sha256",
                "dataset_revision",
                "frame_count",
                "frame_size",
                "feature_dimension",
            )
        },
        "validation_manifest_sha256": _sha256(validation_manifest),
        "forward_cache_sha256": _sha256(forward_cache),
        "forward_report_sha256": _sha256(forward_report),
        "reverse_cache_sha256": _sha256(reverse_cache),
        "reverse_report_sha256": _sha256(reverse_report),
        "validation_scene_count": len(manifest_hashes),
        **result,
    }


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forward-cache", type=Path, required=True)
    parser.add_argument("--forward-report", type=Path, required=True)
    parser.add_argument("--reverse-cache", type=Path, required=True)
    parser.add_argument("--reverse-report", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite feature comparison: {output}")
    report = analyze(
        forward_cache=args.forward_cache.resolve(strict=True),
        forward_report=args.forward_report.resolve(strict=True),
        reverse_cache=args.reverse_cache.resolve(strict=True),
        reverse_report=args.reverse_report.resolve(strict=True),
        validation_manifest=args.validation_manifest.resolve(strict=True),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
