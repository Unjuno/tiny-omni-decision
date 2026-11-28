"""Extract pinned, frozen VideoPrism features from Physion++ TRAIN rows only."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

if __package__:
    from .smoke_physionpp_videoprism_runtime import (
        CHECKPOINT_SHA256,
        FRAME_COUNT,
        FRAME_SIZE,
        MODEL_REVISION,
        SOURCE_COMMIT,
        _read_rows,
        _safe_media_path,
        sha256_file,
        uniform_indices,
    )
else:
    from smoke_physionpp_videoprism_runtime import (
        CHECKPOINT_SHA256,
        FRAME_COUNT,
        FRAME_SIZE,
        MODEL_REVISION,
        SOURCE_COMMIT,
        _read_rows,
        _safe_media_path,
        sha256_file,
        uniform_indices,
    )


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


def select_train_rows(rows: list[dict[str, Any]], limit: int | None = None) -> list[dict[str, Any]]:
    if not rows or any(row.get("split") != "train" for row in rows):
        raise ValueError("feature extraction accepts only non-empty TRAIN rows")
    if any(row.get("source") != "physionpp/readout" for row in rows):
        raise ValueError("manifest contains a non-Physion++ source row")
    if limit is not None and (limit < 1 or limit > len(rows)):
        raise ValueError("limit must be between 1 and the number of train rows")
    selected = rows[:limit] if limit else rows
    if any(len(row.get("media", [])) != 1 for row in selected):
        raise ValueError("each selected row must contain exactly one video asset")
    media_ids = [row["media"][0].get("sha256") for row in selected]
    if any(not isinstance(identity, str) or len(identity) != 64 for identity in media_ids):
        raise ValueError("selected train rows require valid media SHA-256 identities")
    if len(set(media_ids)) != len(media_ids):
        raise ValueError("duplicate media identity in selected train rows")
    return selected


def extract(
    *,
    train_path: Path,
    corpus_root: Path,
    checkpoint: Path,
    source_root: Path,
    output: Path,
    limit: int | None = None,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    if sha256_file(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("VideoPrism checkpoint SHA-256 mismatch")
    import subprocess

    source_commit = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if source_commit != SOURCE_COMMIT:
        raise ValueError(f"unexpected VideoPrism source revision: {source_commit}")

    rows = _read_rows(train_path)
    selected_rows = select_train_rows(rows, limit)
    media_ids = [row["media"][0]["sha256"] for row in selected_rows]
    if len(set(media_ids)) != len(media_ids):
        raise ValueError("duplicate media identity in selected train rows")

    output.mkdir(parents=True, exist_ok=False)
    progress_path = output / "feature-progress.jsonl"
    progress_metadata = output / "run-progress.json"
    started_utc = datetime.now(UTC).isoformat()
    started = time.perf_counter()
    progress: dict[str, Any] = {
        "status": "running_train_only_feature_extraction",
        "train_manifest_sha256": sha256_file(train_path),
        "selected_train_rows": len(selected_rows),
        "completed_rows": 0,
        "created_utc": started_utc,
        "device_policy": "JAX CPU only",
    }
    progress_metadata.write_text(json.dumps(progress, indent=2) + "\n", encoding="utf-8")

    try:
        os.environ["JAX_PLATFORMS"] = "cpu"
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
            for row in selected_rows:
                media = row["media"][0]
                media_path = corpus_root.joinpath(*_safe_media_path(media["path"]).parts)
                if not media_path.is_file() or sha256_file(media_path) != media["sha256"]:
                    raise ValueError(f"missing/hash-mismatched train media: {row['id']}")
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
                if rows_written % 25 == 0 or rows_written == len(selected_rows):
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
                    print(
                        f"completed {rows_written}/{len(selected_rows)} train rows",
                        flush=True,
                    )

        complete = rows_written == len(rows)
        report = {
            "schema_version": 1,
            "status": "complete_train_feature_cache" if complete else "partial_train_feature_cache",
            "claim_scope": (
                "frozen features only; no fitted model, validation score, or quality claim"
            ),
            "run_started_utc": started_utc,
            "run_finished_utc": datetime.now(UTC).isoformat(),
            "source_revision": "0c53b983ca138962ba636c0936e735490ad4c326",
            "train_manifest_sha256": sha256_file(train_path),
            "train_rows_total": len(rows),
            "feature_rows": rows_written,
            "feature_ids_in_manifest_order": [row["id"] for row in selected_rows],
            "feature_media_sha256_in_manifest_order": media_ids,
            "source_commit": source_commit,
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
            "pyav_version": av_version(),
            "rss_before_model_load_bytes": rss_before_model,
            "rss_after_extraction_bytes": process.memory_info().rss,
            "elapsed_seconds": time.perf_counter() - started,
            "feature_progress_jsonl_sha256": sha256_file(progress_path),
        }
        if complete:
            cache_rows = [
                json.loads(line) for line in progress_path.read_text(encoding="utf-8").splitlines()
            ]
            npz_path = output / "video-features.npz"
            np.savez_compressed(
                npz_path,
                sample_ids=np.asarray([item["sample_id"] for item in cache_rows]),
                media_sha256=np.asarray([item["media_sha256"] for item in cache_rows]),
                features=np.asarray([item["feature"] for item in cache_rows], dtype=np.float32),
            )
            report["feature_cache"] = npz_path.name
            report["feature_cache_sha256"] = sha256_file(npz_path)
            report["feature_cache_bytes"] = npz_path.stat().st_size
        (output / "run-report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        progress.update({"status": report["status"], "completed_rows": rows_written})
        progress_metadata.write_text(json.dumps(progress, indent=2) + "\n", encoding="utf-8")
        return report
    except Exception as error:
        progress.update(
            {
                "status": "failed_train_feature_extraction",
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
        progress_metadata.write_text(json.dumps(progress, indent=2) + "\n", encoding="utf-8")
        raise


def av_version() -> str:
    import av

    return av.__version__


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    report = extract(
        train_path=args.train.resolve(strict=True),
        corpus_root=args.corpus_root.resolve(strict=True),
        checkpoint=args.checkpoint.resolve(strict=True),
        source_root=args.source_root.resolve(strict=True),
        output=args.output.resolve(),
        limit=args.limit,
    )
    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key
                not in {"feature_ids_in_manifest_order", "feature_media_sha256_in_manifest_order"}
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
