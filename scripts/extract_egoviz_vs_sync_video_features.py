"""Extract frozen per-frame VideoPrism features for the EgoViz wrist sync pilot."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

CHECKPOINT_SHA256 = "61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29"
UPSTREAM_COMMIT = "c6eba0ece3dd49a24ffc44582966997a4a59f1d8"
EXPECTED_PARAMETERS = 114_365_184


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def temporal_spatial_pool(tokens: np.ndarray, frames: int = 8) -> np.ndarray:
    """Mean-pool spatial tokens but retain one embedding per video frame."""
    if tokens.ndim != 3 or tokens.shape[0] != 1 or tokens.shape[1] % frames:
        raise ValueError("expected VideoPrism tokens with shape [1, frames*spatial, dim]")
    pooled = tokens.reshape(1, frames, tokens.shape[1] // frames, tokens.shape[2]).mean(axis=2)
    result = pooled[0].astype(np.float32)
    if result.shape != (frames, 768) or not np.isfinite(result).all():
        raise ValueError(f"unexpected VideoPrism feature shape/content: {result.shape}")
    return result


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def extract(
    corpus_dir: Path,
    acquisition_manifest: Path,
    checkpoint: Path,
    source_root: Path,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if sha256_file(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("VideoPrism checkpoint hash mismatch")
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source_root, text=True
    ).strip()
    if source_commit != UPSTREAM_COMMIT:
        raise ValueError("VideoPrism upstream source commit mismatch")

    import av
    import jax
    import jax.numpy as jnp
    import psutil
    import tensorflow as tf
    from mcap.reader import make_reader
    from videoprism import models as vp

    from scripts.build_egoviz_vs_sync_window_corpus import VIDEO_TOPIC, _video_message_class
    from scripts.extract_frozen_videoprism_features import _preprocess

    rows = _rows(corpus_dir / "train-samples.jsonl") + _rows(
        corpus_dir / "development-samples.jsonl"
    )
    if not rows or len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("corpus rows are empty or contain duplicate sample IDs")
    inventory = json.loads((corpus_dir / "window-corpus.json").read_text(encoding="utf-8"))
    if inventory["sha256"]["train-samples.jsonl"] != sha256_file(
        corpus_dir / "train-samples.jsonl"
    ) or inventory["sha256"]["development-samples.jsonl"] != sha256_file(
        corpus_dir / "development-samples.jsonl"
    ):
        raise ValueError("window corpus sample manifest hash mismatch")
    acquisition = json.loads(acquisition_manifest.read_text(encoding="utf-8"))
    sources = {item["chunk_id"]: item for item in acquisition["files"]}
    rows_by_chunk: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_chunk.setdefault(row["chunk_id"], []).append(row)

    sys.path.insert(0, str(source_root))
    started = time.perf_counter()
    model = vp.get_model("videoprism_public_v1_base")
    variables = vp.load_pretrained_weights(
        "videoprism_public_v1_base", checkpoint_path=str(checkpoint)
    )
    parameter_count = sum(int(value.size) for value in jax.tree_util.tree_leaves(variables))
    if parameter_count != EXPECTED_PARAMETERS:
        raise ValueError(f"unexpected VideoPrism parameter count: {parameter_count}")
    encode = jax.jit(lambda video: model.apply(variables, video, train=False))
    process = psutil.Process()
    rss_start = process.memory_info().rss
    features: dict[str, np.ndarray] = {}
    per_window_seconds: list[float] = []
    decode_seconds: list[float] = []

    for chunk_id, chunk_rows in rows_by_chunk.items():
        source = sources[chunk_id]
        video_path = Path(source["path"])
        if (
            video_path.stat().st_size != source["bytes"]
            or sha256_file(video_path) != source["sha256"]
        ):
            raise ValueError(f"MCAP size/SHA mismatch: {chunk_id}")
        packets: list[bytes] = []
        video_class = None
        with video_path.open("rb") as stream:
            for schema, _channel, message in make_reader(stream, validate_crcs=True).iter_messages(
                topics=[VIDEO_TOPIC]
            ):
                if schema is None or schema.name != "foxglove.CompressedVideo":
                    raise ValueError("unexpected compressed-video schema")
                if video_class is None:
                    video_class = _video_message_class(schema.data)
                decoded = video_class()
                decoded.ParseFromString(message.data)
                if decoded.format.lower() != "h264":
                    raise ValueError("expected H.264 video")
                packets.append(bytes(decoded.data))
        decode_started = time.perf_counter()
        with av.open(io.BytesIO(b"".join(packets)), format="h264", mode="r") as container:
            frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
        decode_seconds.append(time.perf_counter() - decode_started)
        if len(frames) != len(packets):
            raise ValueError(f"H.264 decoded frame count mismatch: {chunk_id}")

        for row in chunk_rows:
            indices = row["video_frame_indices"]
            if len(indices) != 8 or any(index < 0 or index >= len(frames) for index in indices):
                raise ValueError(f"invalid frame indices: {row['sample_id']}")
            clip = np.stack([frames[index] for index in indices])
            video = jnp.asarray(_preprocess(clip, tf)[None, ...])
            forward_started = time.perf_counter()
            result = encode(video)
            jax.block_until_ready(result)
            per_window_seconds.append(time.perf_counter() - forward_started)
            tokens = np.asarray(result[0] if isinstance(result, tuple) else result)
            features[row["sample_id"]] = temporal_spatial_pool(tokens)

    output.mkdir(parents=True, exist_ok=False)
    ordered = sorted(rows, key=lambda row: row["sample_id"])
    cache_path = output / "video-features.npz"
    np.savez_compressed(
        cache_path,
        sample_id=np.asarray([row["sample_id"] for row in ordered]),
        split=np.asarray([row["split"] for row in ordered]),
        group_id=np.asarray([row["group_id"] for row in ordered]),
        features=np.stack([features[row["sample_id"]] for row in ordered]),
    )
    report = {
        "status": "FROZEN_FEATURES_EXTRACTED_NO_TRAINING_OR_QUALITY_EVALUATION",
        "source_checkpoint_sha256": CHECKPOINT_SHA256,
        "runner_sha256": sha256_file(Path(__file__)),
        "upstream_commit": UPSTREAM_COMMIT,
        "parameter_count": parameter_count,
        "sample_count": len(ordered),
        "train_count": sum(row["split"] == "train" for row in ordered),
        "development_count": sum(row["split"] == "development" for row in ordered),
        "feature_shape": list(np.stack(list(features.values())).shape),
        "feature_semantics": "per-frame spatial-token mean; temporal order retained",
        "feature_file": cache_path.name,
        "feature_file_sha256": sha256_file(cache_path),
        "feature_extraction_seconds": time.perf_counter() - started,
        "video_decode_seconds_by_chunk": decode_seconds,
        "forward_seconds_by_window": per_window_seconds,
        "rss_start_bytes": rss_start,
        "rss_end_bytes": process.memory_info().rss,
        "jax_version": jax.__version__,
        "devices": [str(device) for device in jax.devices()],
        "sealed_audit_accessed": False,
        "training": False,
    }
    (output / "feature-report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--acquisition-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = extract(
        args.corpus,
        args.acquisition_manifest,
        args.checkpoint,
        args.source_root,
        args.output,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
