"""Extract pinned VideoPrism features pooled into STARSS23 azimuth sectors."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from itertools import groupby
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from scripts.extract_frozen_videoprism_features import (
    CHECKPOINT_SHA256,
    SOURCE_COMMIT,
    sha256_file,
)
from scripts.extract_starss23_corpus_features import read_manifest
from scripts.fetch_starss23_frozen_media import write_json
from scripts.starss23_stream import video_windows
from tiny_omni_decision.starss23_spatial_features import panorama_sector_features

MANIFEST_DIR = Path("C:/CodexArtifacts/pretrained-reuse-r2/starss23-all-windows-v1")
CHECKPOINT = Path("C:/CodexArtifacts/pretrained-reuse-r1/videoprism-base-f16r288.npz")
SOURCE_ROOT = Path("C:/CodexArtifacts/pretrained-reuse-r1/videoprism-base-r1/source")
WINDOWS_ROOT = Path("C:/CodexArtifacts/pretrained-reuse-r2")
EXPECTED_FEATURE_ROWS = 2720


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    rows = read_manifest(MANIFEST_DIR)
    if len(rows) != EXPECTED_FEATURE_ROWS:
        raise ValueError(f"expected {EXPECTED_FEATURE_ROWS} windows; found {len(rows)}")
    if sha256_file(CHECKPOINT) != CHECKPOINT_SHA256:
        raise ValueError("pinned VideoPrism checkpoint hash mismatch")
    source_revision = subprocess.check_output(
        ["git", "-C", str(SOURCE_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    if source_revision != SOURCE_COMMIT:
        raise ValueError("pinned VideoPrism source revision mismatch")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    progress_path = args.output_dir / "progress.json"
    report = {
        "status": "RUNNING",
        "modality": "video",
        "task": "eight STARSS23 front-referenced azimuth sector features",
        "manifest_dir": str(MANIFEST_DIR),
        "manifest_result_sha256": sha256_file(MANIFEST_DIR / "result.json"),
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "videoprism_source_commit": SOURCE_COMMIT,
        "runner_sha256": sha256_file(Path(__file__)),
        "feature_runner_sha256": sha256_file(Path("scripts/extract_starss23_corpus_features.py")),
        "spatial_feature_module_sha256": sha256_file(
            Path("src/tiny_omni_decision/starss23_spatial_features.py")
        ),
        "python": sys.version,
        "platform": platform.platform(),
        "device": "cpu",
        "precision": "float32",
        "windows": [],
        "recording_shards": [],
        "training": False,
        "annotation_labels_read": False,
        "dev_test_or_evaluation_or_sealed_audit_media_read": False,
    }
    write_json(progress_path, report)
    try:
        sys.path.insert(0, str(SOURCE_ROOT))
        import jax
        import jax.numpy as jnp
        import tensorflow as tf
        from videoprism import models as vp

        if any(device.platform != "cpu" for device in jax.devices()):
            raise ValueError("this extraction is pinned to CPU")
        tf.config.threading.set_intra_op_parallelism_threads(4)
        tf.config.threading.set_inter_op_parallelism_threads(2)
        model = vp.get_model("videoprism_public_v1_base")
        variables = vp.load_pretrained_weights(
            "videoprism_public_v1_base", checkpoint_path=str(CHECKPOINT)
        )
        forward = jax.jit(lambda x: model.apply(variables, x, train=False))
        report.update(
            parameters=sum(value.size for value in jax.tree_util.tree_leaves(variables)),
            jax=jax.__version__,
            tensorflow=tf.__version__,
            preprocessing="all eight full equirectangular frames resized to 288x288 /255; no crop",
            video_sampling=(
                "existing pinned 8-frame/5-second first-presentation-"
                "at-bin-midpoint policy"
            ),
        )
        vectors = []
        for recording, group in groupby(rows, key=lambda row: row["recording"]):
            group = list(group)
            first_index = len(vectors)
            media = group[0]["media"]["video"]
            if any(row["media"]["video"] != media for row in group):
                raise ValueError("recording maps to inconsistent video media")
            media_path = Path(media["path"])
            if sha256_file(media_path) != media["sha256"]:
                raise ValueError(f"video media hash mismatch: {recording}")
            windows = [(row["start_seconds"], row["end_seconds"]) for row in group]
            for row, (frames, timestamps) in zip(
                group, video_windows(media_path, windows), strict=True
            ):
                tick = time.perf_counter()
                resized = np.asarray(tf.image.resize(frames, (288, 288)), dtype=np.float32) / 255.0
                result = forward(jnp.asarray(resized[None]))
                jax.tree_util.tree_map(lambda value: value.block_until_ready(), result)
                tokens = np.asarray(result[0] if isinstance(result, tuple) else result)
                if tokens.shape != (1, 2048, 768) or not np.isfinite(tokens).all():
                    raise ValueError(f"invalid VideoPrism tokens for {row['id']}")
                feature = panorama_sector_features(tokens, frame_count=8).squeeze(0)
                if feature.shape != (8, 768) or not np.isfinite(feature).all():
                    raise ValueError(f"invalid sector feature for {row['id']}")
                vectors.append(feature)
                report["windows"].append(
                    {
                        "id": row["id"],
                        "recording": recording,
                        "room": row["room"],
                        "split": row["split"],
                        "media_sha256": media["sha256"],
                        "video_presentation_seconds": [float(value) for value in timestamps],
                        "encoder_seconds": time.perf_counter() - tick,
                    }
                )
            shard = np.stack(vectors[first_index:]).astype(np.float32)
            shard_path = args.output_dir / f"recording-{len(report['recording_shards']):03d}.npy"
            np.save(shard_path, shard, allow_pickle=False)
            if not np.array_equal(shard, np.load(shard_path, allow_pickle=False)):
                raise ValueError("recording shard reload mismatch")
            report["recording_shards"].append(
                {
                    "recording": recording,
                    "first_index": first_index,
                    "count": len(group),
                    "path": shard_path.name,
                    "sha256": sha256_file(shard_path),
                }
            )
            write_json(progress_path, report)
            print(
                json.dumps({"recording": recording, "features": len(vectors), "total": len(rows)}),
                flush=True,
            )

        if len(vectors) != len(rows):
            raise ValueError("feature count does not match manifest")
        features = np.stack(vectors).astype(np.float32)
        if features.shape != (len(rows), 8, 768) or not np.isfinite(features).all():
            raise ValueError("feature matrix shape or values are invalid")
        feature_path = args.output_dir / "features.npy"
        np.save(feature_path, features, allow_pickle=False)
        if not np.array_equal(features, np.load(feature_path, allow_pickle=False)):
            raise ValueError("feature matrix reload mismatch")
        report.update(
            status="PASS",
            feature_shape=list(features.shape),
            feature_sha256=sha256_file(feature_path),
            wall_seconds=time.perf_counter() - started,
        )
    except Exception as error:
        report.update(
            status="FAIL",
            error=f"{type(error).__name__}: {error}",
            wall_seconds=time.perf_counter() - started,
        )
        raise
    finally:
        write_json(args.output_dir / "result.json", report)
        write_json(progress_path, report)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
