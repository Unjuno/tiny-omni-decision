"""Center framewise features to preserve a pinned uniform-pooling baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.extract_videoprism_framewise_features import center_framewise_features


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _feature_map(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"media_sha256", "features"}:
            raise ValueError(f"unexpected pooled feature cache fields: {path}")
        hashes = [str(value) for value in archive["media_sha256"].tolist()]
        features = np.asarray(archive["features"], dtype=np.float32)
    if len(hashes) != len(set(hashes)) or features.shape != (len(hashes), 768):
        raise ValueError(f"invalid pooled feature cache shape: {path}")
    if not np.isfinite(features).all():
        raise ValueError(f"non-finite pooled features: {path}")
    return dict(zip(hashes, features, strict=True))


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--train-pooled-cache", type=Path, required=True)
    parser.add_argument("--evaluation-pooled-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _args()
    source_cache = args.source_cache.resolve(strict=True)
    source_report_path = source_cache.with_name("cache-report.json")
    source_report: dict[str, Any] = json.loads(source_report_path.read_text(encoding="utf-8"))
    if source_report.get("status") != "complete_videoprism_framewise_feature_cache":
        raise ValueError("source framewise cache is not complete")
    if _sha256(source_cache) != source_report.get("feature_cache_sha256"):
        raise ValueError("source framewise cache hash mismatch")
    with np.load(source_cache, allow_pickle=False) as archive:
        media_hashes = [str(value) for value in archive["media_sha256"].tolist()]
        features = np.asarray(archive["features"], dtype=np.float32)
    if features.shape != (len(media_hashes), 8, 768) or len(set(media_hashes)) != len(media_hashes):
        raise ValueError("source framewise cache has invalid shape or repeated media")

    train_cache_path = args.train_pooled_cache.resolve(strict=True)
    evaluation_cache_path = args.evaluation_pooled_cache.resolve(strict=True)
    train_pooled = _feature_map(train_cache_path)
    evaluation_pooled = _feature_map(evaluation_cache_path)
    train_assets = set(source_report["train_media_hashes"])
    evaluation_assets = set(source_report["evaluation_media_hashes"])
    if train_assets & evaluation_assets:
        raise ValueError("train/evaluation media overlap in the frozen framewise report")
    if set(media_hashes) != train_assets | evaluation_assets:
        raise ValueError("framewise cache media do not match frozen report identities")
    if not train_assets <= set(train_pooled) or not evaluation_assets <= set(evaluation_pooled):
        raise ValueError("reference pooled cache is missing frozen split assets")

    centered = []
    max_source_delta = 0.0
    max_centered_delta = 0.0
    for media_hash, framewise in zip(media_hashes, features, strict=True):
        reference = (
            train_pooled[media_hash]
            if media_hash in train_assets
            else evaluation_pooled[media_hash]
        )
        max_source_delta = max(
            max_source_delta, float(np.max(np.abs(framewise.mean(axis=0) - reference)))
        )
        corrected, delta = center_framewise_features(framewise, reference)
        max_centered_delta = max(max_centered_delta, delta)
        centered.append(corrected)

    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite centered framewise output: {output}")
    output.mkdir(parents=True)
    cache_path = output / "framewise-features.npz"
    np.savez_compressed(
        cache_path,
        media_sha256=np.asarray(media_hashes),
        features=np.stack(centered),
    )
    report = {
        **source_report,
        "source_feature_cache_sha256": _sha256(source_cache),
        "source_cache_report_sha256": _sha256(source_report_path),
        "centering_correction": (
            "the same per-channel offset was added to all 8 temporal positions for each asset "
            "so its uniform mean reproduces the existing frozen pooled feature"
        ),
        "centering_is_training_or_label_dependent": False,
        "reference_train_pooled_cache_sha256": _sha256(train_cache_path),
        "reference_evaluation_pooled_cache_sha256": _sha256(evaluation_cache_path),
        "max_abs_before_centering": max_source_delta,
        "max_abs_after_centering": max_centered_delta,
        "feature_cache_file": cache_path.name,
        "feature_cache_sha256": _sha256(cache_path),
        "feature_cache_bytes": cache_path.stat().st_size,
    }
    (output / "cache-report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "unique_video_count": len(media_hashes),
                "max_abs_before_centering": max_source_delta,
                "max_abs_after_centering": max_centered_delta,
                "feature_cache_sha256": report["feature_cache_sha256"],
                "feature_cache_bytes": report["feature_cache_bytes"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
