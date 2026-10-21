"""Merge audited frozen VideoPrism caches for the synthetic AV v2 sample IDs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

MODEL_REVISION = "be719a406d563b66f0ac969e7c94bab8e997c81a"
CHECKPOINT_SHA256 = "61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29"
SOURCE_COMMIT = "c6eba0ece3dd49a24ffc44582966997a4a59f1d8"
DATASET_REVISION = "MIT-IBM/CLEVRER@98b842082ba4f7c18b6b9e3f39145871782a65ef"
PREPROCESSING = (
    "uniform endpoint-inclusive frame indices; upstream resize-short-side then center crop; "
    "float32 RGB [0,1]"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_cache(directory: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    report_path = directory / "cache-report.json"
    cache_path = directory / "framewise-features.npz"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("status") != "complete_videoprism_framewise_feature_cache"
        or report.get("dataset_revision") != DATASET_REVISION
        or report.get("model_revision") != MODEL_REVISION
        or report.get("upstream_source_commit") != SOURCE_COMMIT
        or report.get("checkpoint_sha256") != CHECKPOINT_SHA256
        or report.get("frame_count") != 16
        or report.get("feature_dimension") != 768
        or report.get("feature_layout") != "[unique_media, temporal_position, hidden_dimension]"
        or report.get("pooling")
        != "mean over spatial token positions separately at each output frame position"
        or report.get("preprocessing") != PREPROCESSING
        or sha256_file(cache_path) != report.get("feature_cache_sha256")
    ):
        raise ValueError(f"VideoPrism cache provenance or digest mismatch: {directory}")
    with np.load(cache_path, allow_pickle=False) as archive:
        hashes = archive["media_sha256"].astype(str).tolist()
        features = np.asarray(archive["features"], dtype=np.float32)
    if (
        not hashes
        or len(hashes) != len(set(hashes))
        or features.shape != (len(hashes), 16, 768)
        or not np.isfinite(features).all()
    ):
        raise ValueError(f"invalid VideoPrism cache rows: {directory}")
    return dict(zip(hashes, features, strict=True)), report


def merge_features(
    *, sample_dir: Path, source_cache_dirs: list[Path], output_dir: Path
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite feature output: {output_dir}")
    if len(source_cache_dirs) < 2:
        raise ValueError("expected the frozen v1 cache and the expanded official-train cache")
    rows = [
        json.loads(line)
        for split in ("train", "validation")
        for line in (sample_dir / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    if not rows:
        raise ValueError("sample manifests are empty")
    expected = {str(row["media_sha256"]) for row in rows}
    if len(expected) != len({str(row["scene_group_id"]) for row in rows}):
        raise ValueError("one media identity per scene is required")
    combined: dict[str, np.ndarray] = {}
    source_reports = []
    for directory in source_cache_dirs:
        cache, report = _load_cache(directory)
        overlap = set(combined) & set(cache)
        if overlap:
            raise ValueError(f"source VideoPrism caches overlap on {len(overlap)} media hashes")
        combined.update(cache)
        source_reports.append(
            {
                "directory": str(directory.resolve()),
                "cache_sha256": report["feature_cache_sha256"],
                "report_sha256": sha256_file(directory / "cache-report.json"),
                "media_count": len(cache),
            }
        )
    missing = expected - set(combined)
    if missing:
        raise ValueError(f"frozen VideoPrism caches miss {len(missing)} selected media identities")
    selected_hashes = sorted(expected)
    selected_features = np.stack([combined[key] for key in selected_hashes]).astype(
        np.float32, copy=False
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    feature_path = output_dir / "framewise-features.npz"
    np.savez_compressed(
        feature_path,
        media_sha256=np.asarray(selected_hashes),
        features=selected_features,
    )
    report = {
        "status": "complete_videoprism_framewise_feature_cache",
        "cache_scope": "CLEVRER synthetic AV v2 requested media identities only",
        "dataset_revision": DATASET_REVISION,
        "source_split": "official CLEVRER train media only, selected by frozen v2 sample manifests",
        "model_id": "google/videoprism-base-f16r288",
        "model_revision": MODEL_REVISION,
        "upstream_source_commit": SOURCE_COMMIT,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "frame_count": 16,
        "feature_dimension": 768,
        "feature_layout": "[unique_media, temporal_position, hidden_dimension]",
        "pooling": "mean over spatial token positions separately at each output frame position",
        "preprocessing": PREPROCESSING,
        "sample_manifest_sha256": {
            name: sha256_file(sample_dir / name)
            for name in ("train.jsonl", "validation.jsonl", "fetch-report.json")
        },
        "source_caches": source_reports,
        "selected_media_count": len(selected_hashes),
        "selected_media_hashes_sha256": hashlib.sha256(
            "\n".join(selected_hashes).encode()
        ).hexdigest(),
        "feature_cache_file": feature_path.name,
        "feature_cache_sha256": sha256_file(feature_path),
        "feature_cache_bytes": feature_path.stat().st_size,
        "validation_test_or_sealed_data_opened": False,
    }
    (output_dir / "cache-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--source-cache", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = merge_features(
        sample_dir=args.sample_dir.resolve(strict=True),
        source_cache_dirs=[path.resolve(strict=True) for path in args.source_cache],
        output_dir=args.output_dir.resolve(),
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
