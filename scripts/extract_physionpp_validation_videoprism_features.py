"""Extract fixed VideoPrism features from the existing Physion++ DEV validation only."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
for import_root in (REPOSITORY_ROOT, REPOSITORY_ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

if __package__:
    from .audit_physionpp_split_integrity import audit_train_validation_splits
    from .smoke_physionpp_videoprism_runtime import (
        CHECKPOINT_SHA256,
        FRAME_COUNT,
        FRAME_SIZE,
        MODEL_REVISION,
        SOURCE_COMMIT,
        _safe_media_path,
        sha256_file,
        uniform_indices,
    )
else:
    from audit_physionpp_split_integrity import audit_train_validation_splits
    from smoke_physionpp_videoprism_runtime import (
        CHECKPOINT_SHA256,
        FRAME_COUNT,
        FRAME_SIZE,
        MODEL_REVISION,
        SOURCE_COMMIT,
        _safe_media_path,
        sha256_file,
        uniform_indices,
    )


def select_validation_rows(
    rows: list[dict[str, Any]], train_ids: set[str], train_media: set[str]
) -> list[dict[str, Any]]:
    if not rows or any(row.get("split") != "validation" for row in rows):
        raise ValueError("extractor accepts only non-empty development validation rows")
    if any(row.get("source") != "physionpp/readout" for row in rows):
        raise ValueError("validation contains a non-Physion++ source row")
    if any(len(row.get("media", [])) != 1 for row in rows):
        raise ValueError("each validation row must contain exactly one video")
    ids = [row.get("id") for row in rows]
    media_ids = [row["media"][0].get("sha256") for row in rows]
    if len(set(ids)) != len(rows) or len(set(media_ids)) != len(rows):
        raise ValueError("validation sample/media identities must be unique")
    if set(ids) & train_ids or set(media_ids) & train_media:
        raise ValueError("train/validation sample or media identity overlap")
    if any(not isinstance(digest, str) or len(digest) != 64 for digest in media_ids):
        raise ValueError("validation rows require SHA-256 media identities")
    return rows


def _read_validation_rows(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    ids = [row.get("id") for row in rows]
    if (
        not rows
        or any(row.get("split") != "validation" for row in rows)
        or len(set(ids)) != len(rows)
    ):
        raise ValueError("input must be a non-empty, unique development-validation manifest")
    return rows


def _decode(path: Path, expected_frames: int) -> list[np.ndarray]:
    import av

    indices = set(uniform_indices(expected_frames))
    selected = []
    decoded = 0
    with av.open(str(path)) as container:
        stream = next((item for item in container.streams if item.type == "video"), None)
        if stream is None:
            raise ValueError(f"video has no video stream: {path}")
        for index, frame in enumerate(container.decode(stream)):
            decoded = index + 1
            if index in indices:
                selected.append(frame.to_ndarray(format="rgb24"))
    if decoded != expected_frames or len(selected) != len(indices):
        raise ValueError(f"decoded frame count mismatch: {path} ({decoded}/{expected_frames})")
    return selected


def extract_validation(
    *,
    train_path: Path,
    validation_path: Path,
    corpus_root: Path,
    train_feature_cache: Path,
    train_feature_report: Path,
    checkpoint: Path,
    source_root: Path,
    output: Path,
    limit: int | None = None,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    if sha256_file(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("VideoPrism checkpoint SHA-256 mismatch")
    upstream_commit = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if upstream_commit != SOURCE_COMMIT:
        raise ValueError(f"unexpected VideoPrism source revision: {upstream_commit}")

    train = [
        json.loads(line)
        for line in train_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    validation = _read_validation_rows(validation_path)
    if any(row.get("split") != "train" for row in train):
        raise ValueError("train reference manifest contains non-train records")
    train_ids = {row["id"] for row in train}
    train_media = {row["media"][0]["sha256"] for row in train}
    validation = select_validation_rows(validation, train_ids, train_media)
    split_report = audit_train_validation_splits(train_path, validation_path)
    if split_report.get("status") != "disjoint":
        raise ValueError("train/development validation split integrity gate failed")
    if limit is not None and (limit < 1 or limit > len(validation)):
        raise ValueError("limit must be between 1 and the validation row count")
    all_validation_count = len(validation)
    if limit is not None:
        validation = validation[:limit]
    cache_report = json.loads(train_feature_report.read_text(encoding="utf-8"))
    if cache_report.get("status") != "complete_train_feature_cache":
        raise ValueError("pinned train-only VideoPrism cache is incomplete")
    if (
        cache_report.get("model_revision") != MODEL_REVISION
        or cache_report.get("checkpoint_sha256") != CHECKPOINT_SHA256
        or cache_report.get("frame_count") != FRAME_COUNT
        or cache_report.get("frame_size") != FRAME_SIZE
        or cache_report.get("feature_dimension") != 768
        or cache_report.get("device_policy") != "JAX_PLATFORMS=cpu"
    ):
        raise ValueError("train feature cache encoder profile differs from the pinned candidate")
    if cache_report.get("train_manifest_sha256") != sha256_file(train_path):
        raise ValueError("train feature cache references a different train manifest")
    if cache_report.get("feature_cache_sha256") != sha256_file(train_feature_cache):
        raise ValueError("train feature cache hash mismatch")
    if cache_report.get("feature_rows") != len(train):
        raise ValueError("train feature count differs from train rows")

    output.mkdir(parents=True, exist_ok=False)
    progress_path = output / "feature-progress.jsonl"
    started_utc = datetime.now(UTC).isoformat()
    started = time.perf_counter()
    progress_metadata = output / "run-progress.json"
    progress: dict[str, Any] = {
        "status": "running_development_validation_feature_extraction",
        "validation_manifest_sha256": sha256_file(validation_path),
        "validation_rows": len(validation),
        "validation_rows_total": all_validation_count,
        "completed_rows": 0,
        "created_utc": started_utc,
    }
    progress_metadata.write_text(json.dumps(progress, indent=2) + "\n", encoding="utf-8")

    try:
        os.environ["JAX_PLATFORMS"] = "cpu"
        import av
        import jax
        import jax.numpy as jnp
        import psutil
        import tensorflow as tf
        from videoprism import models as vp

        process = psutil.Process()
        rss_before_model = process.memory_info().rss
        model = vp.get_model("videoprism_public_v1_base")
        variables = vp.load_pretrained_weights(
            "videoprism_public_v1_base", checkpoint_path=str(checkpoint)
        )
        encode = jax.jit(lambda video: model.apply(variables, video, train=False))
        rows_written = 0
        with progress_path.open("w", encoding="utf-8", newline="\n") as cache:
            for row in validation:
                media = row["media"][0]
                media_path = corpus_root.joinpath(*_safe_media_path(media["path"]).parts)
                if not media_path.is_file() or sha256_file(media_path) != media["sha256"]:
                    raise ValueError(f"missing/hash-mismatched validation media: {row['id']}")
                frames = _decode(media_path, int(media["num_frames"]))
                height, width = frames[0].shape[:2]
                scale = max(FRAME_SIZE / height, FRAME_SIZE / width)
                resized_shape = (round(height * scale), round(width * scale))
                resized = tf.image.resize(np.stack(frames), resized_shape, method="bilinear")
                prepared = tf.image.resize_with_crop_or_pad(resized, FRAME_SIZE, FRAME_SIZE)
                video = jnp.asarray(np.asarray(prepared, dtype=np.float32)[None, ...] / 255.0)
                result = encode(video)
                jax.tree_util.tree_map(
                    lambda value: (
                        value.block_until_ready() if hasattr(value, "block_until_ready") else value
                    ),
                    result,
                )
                tokens = result[0] if isinstance(result, tuple) else result
                feature = np.asarray(tokens).mean(axis=1).squeeze(0).astype(np.float32)
                if feature.shape != (768,) or not np.isfinite(feature).all():
                    raise ValueError(f"invalid frozen feature for {row['id']}: {feature.shape}")
                cache.write(
                    json.dumps(
                        {
                            "sample_id": row["id"],
                            "media_sha256": media["sha256"],
                            "feature": feature.tolist(),
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                cache.flush()
                rows_written += 1
                if rows_written % 25 == 0 or rows_written == len(validation):
                    progress.update(
                        {
                            "completed_rows": rows_written,
                            "elapsed_seconds": time.perf_counter() - started,
                            "current_rss_bytes": process.memory_info().rss,
                        }
                    )
                    progress_metadata.write_text(
                        json.dumps(progress, indent=2) + "\n", encoding="utf-8"
                    )
                    print(f"completed {rows_written}/{len(validation)} validation rows", flush=True)

        cache_rows = [
            json.loads(line) for line in progress_path.read_text(encoding="utf-8").splitlines()
        ]
        expected_ids = [row["id"] for row in validation]
        expected_media = [row["media"][0]["sha256"] for row in validation]
        if [item["sample_id"] for item in cache_rows] != expected_ids:
            raise ValueError("validation feature IDs/order differ from manifest")
        if [item["media_sha256"] for item in cache_rows] != expected_media:
            raise ValueError("validation feature media identities/order differ from manifest")
        features = np.asarray([item["feature"] for item in cache_rows], dtype=np.float32)
        if features.shape != (len(validation), 768) or not np.isfinite(features).all():
            raise ValueError("validation feature cache shape or finite check failed")
        npz_path = output / "video-features.npz"
        np.savez_compressed(
            npz_path,
            sample_ids=np.asarray(expected_ids),
            media_sha256=np.asarray(expected_media),
            features=features,
        )
        report = {
            "schema_version": 1,
            "status": (
                "complete_development_validation_feature_cache"
                if len(validation) == all_validation_count
                else "partial_development_validation_feature_cache"
            ),
            "claim_scope": (
                "frozen development-validation features only; not blind or sealed evaluation"
            ),
            "run_started_utc": started_utc,
            "run_finished_utc": datetime.now(UTC).isoformat(),
            "pipeline_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
            "extractor_sha256": sha256_file(Path(__file__).resolve()),
            "train_manifest_sha256": sha256_file(train_path),
            "validation_manifest_sha256": sha256_file(validation_path),
            "split_integrity": split_report,
            "train_feature_cache_sha256": sha256_file(train_feature_cache),
            "train_feature_report_sha256": sha256_file(train_feature_report),
            "validation_rows": len(validation),
            "validation_rows_total": all_validation_count,
            "validation_ids_in_manifest_order": expected_ids,
            "validation_media_sha256_in_manifest_order": expected_media,
            "source_commit": upstream_commit,
            "model_revision": MODEL_REVISION,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "frame_count": FRAME_COUNT,
            "frame_size": FRAME_SIZE,
            "feature_dimension": 768,
            "pooling": "mean across VideoPrism spatiotemporal tokens",
            "device_policy": "JAX_PLATFORMS=cpu",
            "jax_devices": [str(device) for device in jax.devices()],
            "jax_version": jax.__version__,
            "tensorflow_version": tf.__version__,
            "pyav_version": av.__version__,
            "rss_before_model_load_bytes": rss_before_model,
            "rss_after_extraction_bytes": process.memory_info().rss,
            "elapsed_seconds": time.perf_counter() - started,
            "feature_progress_jsonl_sha256": sha256_file(progress_path),
            "feature_cache_sha256": sha256_file(npz_path),
            "feature_cache_bytes": npz_path.stat().st_size,
        }
        (output / "run-report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        progress.update({"status": report["status"], "completed_rows": rows_written})
        progress_metadata.write_text(json.dumps(progress, indent=2) + "\n", encoding="utf-8")
        return report
    except Exception as error:
        progress.update(
            {
                "status": "failed_validation_feature_extraction",
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
        progress_metadata.write_text(json.dumps(progress, indent=2) + "\n", encoding="utf-8")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--train-feature-cache", type=Path, required=True)
    parser.add_argument("--train-feature-report", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    report = extract_validation(
        train_path=args.train.resolve(strict=True),
        validation_path=args.validation.resolve(strict=True),
        corpus_root=args.corpus_root.resolve(strict=True),
        train_feature_cache=args.train_feature_cache.resolve(strict=True),
        train_feature_report=args.train_feature_report.resolve(strict=True),
        checkpoint=args.checkpoint.resolve(strict=True),
        source_root=args.source_root.resolve(strict=True),
        output=args.output.resolve(),
        limit=args.limit,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
