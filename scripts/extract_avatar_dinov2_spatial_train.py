"""Extract aspect-preserving DINOv2 spatial tokens for a TRAIN-only AVATAR fold."""

from __future__ import annotations

import argparse
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

from scripts.extract_avatar_dinov2_train_frames import (  # noqa: E402
    MODEL_REVISION,
    MODEL_SHA256,
    _decode_clip_targets,
    read_jsonl,
    sha256_file,
    write_json_atomic,
)


def letterbox_frame(
    frame: np.ndarray, *, width: int = 224, height: int = 140
) -> tuple[np.ndarray, dict[str, float]]:
    from PIL import Image

    image = Image.fromarray(frame)
    original_width, original_height = image.size
    scale = min(width / original_width, height / original_height)
    resized_width = max(1, round(original_width * scale))
    resized_height = max(1, round(original_height * scale))
    resized = image.resize((resized_width, resized_height), Image.Resampling.BICUBIC)
    left = (width - resized_width) // 2
    top = (height - resized_height) // 2
    canvas = np.asarray([124, 116, 104], dtype=np.uint8).reshape(1, 1, 3)
    canvas = np.broadcast_to(canvas, (height, width, 3)).copy()
    canvas[top : top + resized_height, left : left + resized_width] = np.asarray(resized)
    transform = {
        "scale_x": resized_width / original_width,
        "scale_y": resized_height / original_height,
        "pad_x": float(left),
        "pad_y": float(top),
        "canvas_width": float(width),
        "canvas_height": float(height),
    }
    return canvas, transform


def bbox_center_cell(bbox_xywhs: list[list[float]], transform: dict[str, float]) -> list[int]:
    """Map all annotated bbox centers to the fixed 8x5 DINO patch-cell grid."""
    cells = set()
    for x, y, width, height in bbox_xywhs:
        center_x = transform["pad_x"] + (x + width / 2) * transform["scale_x"]
        center_y = transform["pad_y"] + (y + height / 2) * transform["scale_y"]
        col = min(7, max(0, int(center_x / 28)))
        row = min(4, max(0, int(center_y / 28)))
        cells.add(row * 8 + col)
    if not cells:
        raise ValueError("at least one bbox is required for localization")
    return sorted(cells)


def _load_assets(paths: list[Path]) -> list[dict[str, Any]]:
    by_key: dict[str, dict[str, Any]] = {}
    for path in paths:
        for line_number, row in enumerate(read_jsonl(path), 1):
            key = row.get("asset_key")
            timestamp = row.get("timestamp_seconds")
            if not isinstance(key, str) or key.count("/") != 1:
                raise ValueError(f"{path}:{line_number}: malformed asset_key")
            if not isinstance(timestamp, (int, float)) or not np.isfinite(timestamp):
                raise ValueError(f"{path}:{line_number}: malformed timestamp_seconds")
            if key in by_key:
                raise ValueError(f"duplicate feature asset key: {key}")
            by_key[key] = {"asset_key": key, "timestamp_seconds": float(timestamp)}
    return [by_key[key] for key in sorted(by_key)]


def extract(args: argparse.Namespace) -> dict[str, Any]:
    train_path = args.train_input.resolve(strict=True)
    validation_path = args.validation_input.resolve(strict=True)
    archive_path = args.video_zip.resolve(strict=True)
    model_dir = args.model_dir.resolve(strict=True)
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite feature cache: {output_path}")
    model_path = model_dir / "model.safetensors"
    if sha256_file(model_path) != MODEL_SHA256:
        raise ValueError("local DINOv2-base checkpoint hash differs from the pinned model")

    assets = _load_assets([train_path, validation_path])
    by_clip: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for asset in assets:
        clip_id, _frame_id = asset["asset_key"].split("/", 1)
        by_clip[clip_id].append(asset)
    if args.limit_clips is not None:
        if args.limit_clips < 1:
            raise ValueError("limit-clips must be positive")
        selected_clips = set(sorted(by_clip)[: args.limit_clips])
        by_clip = defaultdict(
            list, {clip: rows for clip, rows in by_clip.items() if clip in selected_clips}
        )
        assets = [
            asset for asset in assets if asset["asset_key"].split("/", 1)[0] in selected_clips
        ]

    import av
    import torch
    import torch.nn.functional as functional

    device = torch.device(args.device)
    from transformers import AutoImageProcessor, Dinov2Model

    processor = AutoImageProcessor.from_pretrained(model_dir, local_files_only=True, use_fast=False)
    mean = torch.tensor(processor.image_mean, device=device).view(1, 3, 1, 1)
    std = torch.tensor(processor.image_std, device=device).view(1, 3, 1, 1)
    model = Dinov2Model.from_pretrained(model_dir, local_files_only=True).eval().to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    pending_images: list[np.ndarray] = []
    pending_assets: list[dict[str, Any]] = []
    features: dict[str, np.ndarray] = {}
    frame_errors: dict[str, float] = {}
    started = time.perf_counter()
    processed_clips = 0

    def encode_pending() -> None:
        if not pending_images:
            return
        arrays = np.stack(pending_images)
        pixels = torch.from_numpy(arrays).to(device=device, dtype=torch.float32).permute(0, 3, 1, 2)
        pixels = (pixels / 255.0 - mean) / std
        with torch.inference_mode():
            output = model(pixel_values=pixels, interpolate_pos_encoding=True)
            patch_tokens = output.last_hidden_state[:, 1:, :]
            batch = patch_tokens.shape[0]
            patch_grid = patch_tokens.transpose(1, 2).reshape(batch, 768, 10, 16)
            pooled = functional.adaptive_avg_pool2d(patch_grid, (5, 8)).permute(0, 2, 3, 1)
            values = pooled.to(dtype=torch.float16).cpu().numpy()
        if values.shape != (len(pending_assets), 5, 8, 768) or not np.isfinite(values).all():
            raise ValueError("DINOv2 spatial features have an invalid shape or non-finite values")
        for asset, value in zip(pending_assets, values, strict=True):
            key = asset["asset_key"]
            features[key] = value
            frame_errors[key] = float(asset["frame_error_seconds"])
        pending_images.clear()
        pending_assets.clear()

    with zipfile.ZipFile(archive_path) as archive:
        archive_names = set(archive.namelist())
        for clip_id in sorted(by_clip):
            member = f"video/{clip_id}.mp4"
            if member not in archive_names:
                raise FileNotFoundError(f"TRAIN video missing from media archive: {member}")
            targets = by_clip[clip_id]
            frames, errors = _decode_clip_targets(
                archive.read(member),
                [asset["timestamp_seconds"] for asset in targets],
                av_module=av,
            )
            for asset, frame, error in zip(targets, frames, errors, strict=True):
                image, transform = letterbox_frame(frame)
                asset["frame_error_seconds"] = error
                asset["letterbox_transform"] = transform
                pending_assets.append(asset)
                pending_images.append(image)
                if len(pending_images) >= args.batch_size:
                    encode_pending()
            processed_clips += 1
            if processed_clips % 100 == 0:
                print(
                    json.dumps(
                        {
                            "processed_clips": processed_clips,
                            "total_clips": len(by_clip),
                            "encoded_frames": len(features),
                            "total_frames": len(assets),
                            "elapsed_seconds": round(time.perf_counter() - started, 1),
                        }
                    ),
                    flush=True,
                )
        encode_pending()

    keys = [asset["asset_key"] for asset in assets]
    if set(features) != set(keys):
        raise ValueError("spatial DINO features do not match the TRAIN-only fold assets")
    ordered = sorted(assets, key=lambda row: row["asset_key"])
    feature_array = np.stack([features[row["asset_key"]] for row in ordered])
    timestamp_array = np.asarray([row["timestamp_seconds"] for row in ordered], dtype=np.float64)
    error_array = np.asarray([frame_errors[row["asset_key"]] for row in ordered], dtype=np.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp.npz")
    np.savez_compressed(
        temporary,
        asset_keys=np.asarray(keys, dtype=np.str_),
        features=feature_array,
        timestamps_seconds=timestamp_array,
        frame_error_seconds=error_array,
    )
    temporary.replace(output_path)
    report = {
        "experiment_id": "avatar-dinov2-spatial-localization-train-only-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "source_revision": "f32d1755deaaabd8cfb74a5558696b4d6808ae68",
        "model_id": "facebook/dinov2-base",
        "model_revision": MODEL_REVISION,
        "model_sha256": MODEL_SHA256,
        "train_input_sha256": sha256_file(train_path),
        "validation_input_sha256": sha256_file(validation_path),
        "video_zip_sha256": sha256_file(archive_path),
        "feature_cache_sha256": sha256_file(output_path),
        "feature_cache_bytes": output_path.stat().st_size,
        "feature_count": len(keys),
        "feature_shape_per_asset": [5, 8, 768],
        "clip_count": len(by_clip),
        "frame_time_error_max_seconds": float(error_array.max(initial=0.0)),
        "frame_time_error_mean_seconds": float(error_array.mean()),
        "device": str(device),
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else None,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
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
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit-clips", type=int)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    print(json.dumps(extract(args), indent=2), flush=True)


if __name__ == "__main__":
    main()
