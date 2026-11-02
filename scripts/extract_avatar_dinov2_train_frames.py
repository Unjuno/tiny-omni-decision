"""Extract pinned DINOv2-base CLS features for a frozen AVATAR TRAIN-only fold."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
import zipfile
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

MODEL_REVISION = "f9e44c814b77203eaa57a6bdbbd535f21ede1415"
MODEL_SHA256 = "d73036b56966966d07975d696bde331762f37297e2f095de8cea0040c3aa0841"
SOURCE_REVISION = "a7f0146ff704799d044133e61137836cacc1c233"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError(f"empty JSONL: {path}")
    return rows


def write_json_atomic(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def expected_video_assets(
    input_paths: list[Path], *, limit_assets: int | None = None
) -> list[dict[str, Any]]:
    by_key: dict[str, float] = {}
    for path in input_paths:
        for line_number, row in enumerate(read_jsonl(path), 1):
            if row.get("split") not in {"train", "validation"}:
                raise ValueError(f"{path}:{line_number}: unexpected split")
            for option in row.get("options", []):
                key = option.get("video_asset_key")
                timestamp = option.get("video_timestamp_seconds")
                if not isinstance(key, str) or key.count("/") != 1:
                    raise ValueError(f"{path}:{line_number}: malformed video asset key")
                if not isinstance(timestamp, (int, float)) or not np.isfinite(timestamp):
                    raise ValueError(f"{path}:{line_number}: malformed video timestamp")
                previous = by_key.setdefault(key, float(timestamp))
                if abs(previous - float(timestamp)) > 1e-9:
                    raise ValueError(f"conflicting timestamp for {key}")
    keys = sorted(by_key)
    if limit_assets is not None:
        if limit_assets < 1:
            raise ValueError("limit_assets must be positive")
        keys = keys[:limit_assets]
    return [{"asset_key": key, "timestamp_seconds": by_key[key]} for key in keys]


def _decode_clip_targets(
    clip_bytes: bytes, targets: list[float], *, av_module: Any
) -> tuple[list[np.ndarray], list[float]]:
    with av_module.open(io.BytesIO(clip_bytes), mode="r") as container:
        streams = [stream for stream in container.streams if stream.type == "video"]
        if not streams:
            raise ValueError("video archive member has no video stream")
        stream = streams[0]
        if not targets:
            return [], []
        duration = (
            float(container.duration / av_module.time_base)
            if container.duration is not None
            else 0.0
        )
        if duration <= 0:
            raise ValueError("video duration unavailable")
        requested = np.clip(np.asarray(targets, dtype=np.float64), 0.0, duration - 1e-6)
        container.seek(
            max(0, int(float(requested.min()) / float(stream.time_base))),
            stream=stream,
            backward=True,
            any_frame=False,
        )
        frames: list[np.ndarray] = []
        times: list[float] = []
        last_target = float(requested.max())
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            frames.append(frame.to_ndarray(format="rgb24"))
            times.append(float(frame.pts * stream.time_base))
            if times[-1] > last_target + 0.1:
                break
        if not frames:
            raise ValueError("no frame decoded near requested timestamp")
        decoded_times = np.asarray(times)
        indices = np.abs(requested[:, None] - decoded_times[None, :]).argmin(axis=1)
        selected = [frames[int(index)] for index in indices]
        errors = [
            float(abs(decoded_times[index] - target))
            for index, target in zip(indices, requested, strict=True)
        ]
        return selected, errors


def extract(args: argparse.Namespace) -> dict[str, Any]:
    input_paths = [
        args.train_input.resolve(strict=True),
        args.validation_input.resolve(strict=True),
    ]
    archive_path = args.video_zip.resolve(strict=True)
    model_dir = args.model_dir.resolve(strict=True)
    model_path = model_dir / "model.safetensors"
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing feature cache: {output_path}")
    if sha256_file(model_path) != MODEL_SHA256:
        raise ValueError("local DINOv2-base checkpoint hash differs from the pinned model")

    assets = expected_video_assets(input_paths, limit_assets=args.limit_assets)
    by_clip: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for asset in assets:
        key = str(asset["asset_key"])
        clip_id, frame_id = key.split("/", 1)
        by_clip[clip_id].append({**asset, "frame_id": frame_id})

    import av
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, Dinov2Model

    device = torch.device(args.device)
    processor = AutoImageProcessor.from_pretrained(model_dir, local_files_only=True, use_fast=False)
    model = Dinov2Model.from_pretrained(model_dir, local_files_only=True).eval().to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    batch_images: list[np.ndarray] = []
    batch_assets: list[dict[str, Any]] = []
    results: dict[str, tuple[np.ndarray, float]] = {}
    started = time.perf_counter()
    processed_clips = 0

    def encode_pending() -> None:
        if not batch_images:
            return
        pil_images = [Image.fromarray(image) for image in batch_images]
        encoded = processor(images=pil_images, return_tensors="pt")
        pixel_values = encoded["pixel_values"].to(device)
        with torch.inference_mode():
            output = model(pixel_values=pixel_values)
            vectors = output.pooler_output.float().cpu().numpy()
        if vectors.shape != (len(batch_assets), 768) or not np.isfinite(vectors).all():
            raise ValueError("DINOv2 returned invalid CLS feature vectors")
        for asset, vector in zip(batch_assets, vectors, strict=True):
            results[str(asset["asset_key"])] = (vector, float(asset["frame_error"]))
        batch_images.clear()
        batch_assets.clear()

    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        for clip_id in sorted(by_clip):
            member = f"video/{clip_id}.mp4"
            if member not in names:
                raise FileNotFoundError(f"video member missing from pinned archive: {member}")
            targets = by_clip[clip_id]
            frames, errors = _decode_clip_targets(
                archive.read(member),
                [float(asset["timestamp_seconds"]) for asset in targets],
                av_module=av,
            )
            for asset, frame, error in zip(targets, frames, errors, strict=True):
                asset["frame_error"] = error
                batch_assets.append(asset)
                batch_images.append(frame)
                if len(batch_images) >= args.batch_size:
                    encode_pending()
            processed_clips += 1
            if processed_clips % 100 == 0:
                print(
                    json.dumps(
                        {
                            "clips": processed_clips,
                            "assets_encoded": len(results),
                            "assets_total": len(assets),
                            "elapsed_seconds": round(time.perf_counter() - started, 1),
                        }
                    ),
                    flush=True,
                )
        encode_pending()

    expected_keys = {str(asset["asset_key"]) for asset in assets}
    if set(results) != expected_keys:
        raise ValueError("DINOv2 feature IDs differ from the frozen TRAIN fold assets")
    ordered = sorted(assets, key=lambda item: str(item["asset_key"]))
    features = np.stack([results[str(item["asset_key"])][0] for item in ordered]).astype(
        np.float32, copy=False
    )
    timestamps = np.asarray([item["timestamp_seconds"] for item in ordered], dtype=np.float64)
    errors = np.asarray([results[str(item["asset_key"])][1] for item in ordered], dtype=np.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp.npz")
    np.savez_compressed(
        temporary,
        asset_keys=np.asarray([item["asset_key"] for item in ordered]),
        features=features,
        timestamps=timestamps,
        max_frame_time_error_seconds=errors,
    )
    temporary.replace(output_path)
    peak_memory = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    report = {
        "experiment_id": "avatar-dinov2-framewise-train-fold-v1",
        "source_revision": SOURCE_REVISION,
        "model_id": "facebook/dinov2-base",
        "model_revision": MODEL_REVISION,
        "model_sha256": MODEL_SHA256,
        "video_zip_sha256": sha256_file(archive_path),
        "video_zip_bytes": archive_path.stat().st_size,
        "input_sha256": {str(path): sha256_file(path) for path in input_paths},
        "feature_cache": str(output_path),
        "feature_cache_sha256": sha256_file(output_path),
        "feature_count": len(results),
        "feature_dimension": int(features.shape[1]),
        "clip_count": len(by_clip),
        "frame_error_max_seconds": float(errors.max(initial=0.0)),
        "frame_error_mean_seconds": float(errors.mean()),
        "device": str(device),
        "peak_cuda_allocated_bytes": peak_memory,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "created_utc": datetime.now(UTC).isoformat(),
        "sealed_audit_accessed": False,
        "media_written_to_output": False,
    }
    write_json_atomic(output_path.with_suffix(".metadata.json"), report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-input", required=True, type=Path)
    parser.add_argument("--validation-input", required=True, type=Path)
    parser.add_argument("--video-zip", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit-assets", type=int)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    print(json.dumps(extract(args), indent=2), flush=True)


if __name__ == "__main__":
    main()
