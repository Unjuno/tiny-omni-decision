"""Extract pinned VideoPrism features from train or development video windows."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from scripts.eidon_range_io import HTTPRangeReader
from scripts.extract_frozen_videoprism_features import (
    CHECKPOINT_SHA256,
    FRAME_COUNT,
    MODEL_REVISION,
    SOURCE_COMMIT,
    _preprocess,
    sha256_file,
)
from scripts.extract_videoprism_framewise_features import pool_framewise_tokens

VIDEO_REVISION = "47c55ccc6d0308894ab482009c6fa91569057a08"
FRAME_WINDOW_SECONDS = 5.0
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""


def target_times(duration_seconds: float, count: int = FRAME_COUNT,
                 window_seconds: float = FRAME_WINDOW_SECONDS) -> np.ndarray:
    if (not np.isfinite(duration_seconds) or duration_seconds < window_seconds
            or isinstance(count, bool) or count < 1):
        raise ValueError("video must contain the requested finite frame window")
    start = (duration_seconds - window_seconds) / 2
    return np.linspace(start, start + window_seconds, count, dtype=np.float64)


def _decode_window(reader: HTTPRangeReader, duration_seconds: float) -> np.ndarray:
    import av

    targets = target_times(duration_seconds)
    container = av.open(reader, format="mp4")
    try:
        streams = [stream for stream in container.streams if stream.type == "video"]
        if not streams:
            raise ValueError("video has no video stream")
        stream = streams[0]
        actual_duration = (
            float(stream.duration * stream.time_base)
            if stream.duration is not None else duration_seconds
        )
        if abs(actual_duration - duration_seconds) > 2.0:
            raise ValueError("video stream duration disagrees with frozen metadata")
        start = float(targets[0])
        container.seek(int(start / float(stream.time_base)), stream=stream,
                       backward=True, any_frame=False)
        selected = []
        target_index = 0
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            timestamp = float(frame.pts * frame.time_base)
            if timestamp < start:
                continue
            if timestamp > targets[-1] + 0.5:
                break
            while target_index < len(targets) and timestamp >= targets[target_index]:
                selected.append(frame.to_ndarray(format="rgb24"))
                target_index += 1
        if len(selected) != FRAME_COUNT:
            raise ValueError(f"decoder produced {len(selected)}/{FRAME_COUNT} window frames")
        return np.stack(selected)
    finally:
        container.close()


def extract_split(manifest_path: Path, sensor_records_path: Path, checkpoint: Path,
                  source_root: Path, split: str, output_dir: Path) -> dict:
    if split not in {"train", "development"}:
        raise ValueError("only train and development are allowed")
    if manifest_path.name != "train-development-manifest-private.json":
        raise ValueError("only the frozen train/development manifest is allowed")
    if not sensor_records_path.is_file():
        raise FileNotFoundError("validated sensor record manifest is required")
    sensor_cache_path = sensor_records_path.with_name("features.npz")
    sensor_report_path = sensor_records_path.with_name("result.json")
    if not sensor_cache_path.is_file() or not sensor_report_path.is_file():
        raise ValueError("completed validated sensor feature cache is required")
    sensor_report = json.loads(sensor_report_path.read_text(encoding="utf-8"))
    if (sensor_report.get("split") != split
            or sensor_report.get("records_sha256") != sha256_file(sensor_records_path)
            or sensor_report.get("feature_sha256") != sha256_file(sensor_cache_path)):
        raise ValueError("validated sensor feature cache hash or split mismatch")
    if sha256_file(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("pinned VideoPrism checkpoint hash mismatch")
    upstream = subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if upstream != SOURCE_COMMIT:
        raise ValueError("pinned VideoPrism source revision mismatch")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("video_revision") != VIDEO_REVISION:
        raise ValueError("unexpected Eidon video revision")
    rows = {int(row["recording_id"]): row for row in manifest[split]}
    accepted = json.loads(sensor_records_path.read_text(encoding="utf-8"))
    selected = []
    for record in accepted:
        recording_id = int(record["recording_id"])
        row = rows.get(recording_id)
        if row is None or record.get("split") != split:
            raise ValueError("sensor/video split identities differ")
        if record.get("asset_sha256") != row.get("asset_sha256"):
            raise ValueError("sensor/video media identities differ")
        selected.append(row)
    if not selected or len({row["recording_id"] for row in selected}) != len(selected):
        raise ValueError("empty or duplicate validated video records")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")

    sys.path.insert(0, str(source_root))
    import jax
    import jax.numpy as jnp
    import tensorflow as tf
    from videoprism import models as vp

    model = vp.get_model("videoprism_public_v1_base")
    variables = vp.load_pretrained_weights(
        "videoprism_public_v1_base", checkpoint_path=str(checkpoint)
    )
    encode = jax.jit(lambda video: model.apply(variables, video, train=False))
    output_dir.mkdir(parents=True, exist_ok=False)
    features, details, stats = [], [], []
    started = time.perf_counter()
    for index, row in enumerate(selected, start=1):
        url = (f"https://huggingface.co/datasets/eidon-ai/tracker-pov/resolve/"
               f"{VIDEO_REVISION}/{row['file_name']}")
        video_started = time.perf_counter()
        with HTTPRangeReader(url, max_bytes=128 * 1024 * 1024,
                             block_size=8 * 1024 * 1024, max_cached_blocks=6) as source:
            frames = _decode_window(source, float(row["video_duration_seconds"]))
            video = jnp.asarray(_preprocess(frames, tf)[None, ...])
            result = encode(video)
            jax.tree_util.tree_map(
                lambda value: (value.block_until_ready()
                               if hasattr(value, "block_until_ready") else value), result
            )
            tokens = result[0] if isinstance(result, tuple) else result
            framewise = pool_framewise_tokens(np.asarray(tokens), FRAME_COUNT)
            if framewise.shape != (FRAME_COUNT, 768):
                raise ValueError("unexpected framewise VideoPrism features")
            features.append(framewise)
            stats.append({
                "recording_id": int(row["recording_id"]),
                "bytes_fetched": source.bytes_read,
                "range_requests": source.range_requests,
                "elapsed_seconds": round(time.perf_counter() - video_started, 3),
            })
        details.append({
            "recording_id": int(row["recording_id"]),
            "asset_sha256": row["asset_sha256"],
            "contributor_id": row["contributor_id"],
            "task_type": row["task_type"],
            "split": split,
        })
        print(f"{split} {index}/{len(selected)}", flush=True)

    output = np.stack(features).astype(np.float32)
    if output.shape != (len(selected), FRAME_COUNT, 768) or not np.isfinite(output).all():
        raise ValueError("non-finite or unexpected Eidon VideoPrism feature tensor")
    cache_path = output_dir / "features.npz"
    np.savez_compressed(
        cache_path,
        recording_ids=np.asarray([row["recording_id"] for row in details], dtype=np.int64),
        asset_sha256=np.asarray([row["asset_sha256"] for row in details]),
        values=output,
    )
    detail_path = output_dir / "records.json"
    detail_path.write_text(json.dumps(details, indent=2) + "\n", encoding="utf-8")
    report = {
        "status": "complete_frozen_videoprism_features",
        "split": split,
        "count": len(selected),
        "shape": list(output.shape),
        "dataset_revision": VIDEO_REVISION,
        "model_revision": MODEL_REVISION,
        "upstream_source_commit": upstream,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "manifest_sha256": sha256_file(manifest_path),
        "sensor_feature_sha256": sensor_report["feature_sha256"],
        "sensor_records_sha256": sensor_report["records_sha256"],
        "frame_count": FRAME_COUNT,
        "frame_size": 288,
        "window_seconds": FRAME_WINDOW_SECONDS,
        "pooling": "mean across 256 spatial tokens separately for each of 8 frames",
        "preprocessing": "upstream resize-short-side then center crop; float32 RGB [0,1]",
        "feature_sha256": sha256_file(cache_path),
        "records_sha256": sha256_file(detail_path),
        "sample_stats": stats,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "devices": [str(device) for device in jax.devices()],
        "jax_version": jax.__version__,
        "tensorflow_version": tf.__version__,
        "sealed_audit_read": False,
    }
    (output_dir / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sensor-records", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "development"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = extract_split(args.manifest, args.sensor_records, args.checkpoint,
                           args.source_root, args.split, args.output_dir)
    print(json.dumps({key: report[key] for key in (
        "split", "count", "shape", "feature_sha256", "elapsed_seconds", "devices"
    )}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
