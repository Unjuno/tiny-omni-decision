"""Extract unique frozen DINOv2 frame features for frozen sub-SEVIR windows."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from tiny_omni_decision.subsevir_features import (
    cls_plus_patch_mean,
    frame_to_rgb,
    required_frame_indices,
    train_event_percentile_ranges,
)

MODEL_REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
MODEL_SHA256 = "ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1"
EXPECTED_PARAMETERS = 22_056_576


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_model(model_dir: Path, torch: Any, transformers: Any) -> tuple[Any, Any]:
    weight_file = model_dir / "model.safetensors"
    if _file_sha256(weight_file) != MODEL_SHA256:
        raise ValueError("DINOv2 checkpoint hash mismatch")
    processor = transformers.AutoImageProcessor.from_pretrained(
        model_dir, local_files_only=True
    )
    model, loading = transformers.AutoModel.from_pretrained(
        model_dir, local_files_only=True, output_loading_info=True
    )
    error_keys = ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    if any(loading.get(key) for key in error_keys):
        raise ValueError(f"DINOv2 did not load strictly: {loading}")
    count = sum(parameter.numel() for parameter in model.parameters())
    if count != EXPECTED_PARAMETERS:
        raise ValueError(f"unexpected DINOv2 parameter count: {count}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this local extraction")
    device = torch.device("cuda")
    model.to(device).eval().requires_grad_(False)
    return processor, model


def _encode_rows(
    row_indices: list[int],
    group: Any,
    ranges: list[list[float]],
    processor: Any,
    model: Any,
    torch: Any,
    image_cls: Any,
) -> np.ndarray:
    raw_rows = np.asarray(group["features"][row_indices], dtype=np.float32)
    images = []
    for frame in raw_rows:
        for stream in ("satellite", "radar"):
            images.append(image_cls.fromarray(frame_to_rgb(frame, ranges, stream), mode="RGB"))
    batch = processor(images=images, return_tensors="pt")["pixel_values"].to("cuda")
    with torch.inference_mode():
        tokens = model(pixel_values=batch).last_hidden_state
    features = cls_plus_patch_mean(tokens.float().cpu().numpy())
    if features.shape != (len(row_indices) * 2, 768):
        raise ValueError(f"unexpected DINOv2 batch features: {features.shape}")
    return features.reshape(len(row_indices), 2, 768)


def _run_dry_run(args: argparse.Namespace, manifest: dict[str, object]) -> dict[str, object]:
    if args.dry_run_report.exists():
        raise FileExistsError(f"refusing to overwrite throughput report: {args.dry_run_report}")
    sys.path.append(str(args.zarr_site_packages))
    import torch
    import transformers
    import zarr
    from PIL import Image

    group = zarr.open_group(str(args.train_zarr), mode="r")
    train_groups = set(manifest["split_group_ids"]["train"])
    ranges = train_event_percentile_ranges(group, train_groups)
    split_indices = required_frame_indices(manifest["windows"])
    train_indices = split_indices["train"]
    processor, model = _load_model(args.model_dir, torch, transformers)
    batch_rows = max(1, args.batch_images // 2)
    process = torch.cuda.current_device()
    torch.cuda.reset_peak_memory_stats(process)
    started = time.perf_counter()
    processed = 0
    batches = 0
    cursor = 0
    first = True
    while time.perf_counter() - started < args.dry_run_seconds:
        indices = train_indices[cursor : cursor + batch_rows]
        if not indices:
            cursor = 0
            continue
        _encode_rows(indices, group, ranges, processor, model, torch, Image)
        if first:
            torch.cuda.synchronize()
            first = False
        processed += len(indices) * 2
        batches += 1
        cursor += len(indices)
        if cursor >= len(train_indices):
            cursor = 0
        if batches % 500 == 0:
            elapsed = time.perf_counter() - started
            print(json.dumps({
                "event": "throughput_progress",
                "elapsed_seconds": round(elapsed, 1),
                "encoded_stream_frames": processed,
                "stream_frames_per_second": processed / elapsed,
            }), flush=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    total_stream_frames = sum(len(value) for value in split_indices.values()) * 2
    report = {
        "schema": "tiny-omni-decision/subsevir-dinov2-throughput-v1",
        "scope": "train-only repeated throughput dry run; no feature cache or validation access",
        "manifest_sha256": _file_sha256(args.manifest),
        "model_revision": MODEL_REVISION,
        "weights_sha256": MODEL_SHA256,
        "device": torch.cuda.get_device_name(process),
        "batch_images": batch_rows * 2,
        "dry_run_seconds": elapsed,
        "batches": batches,
        "encoded_stream_frames": processed,
        "stream_frames_per_second": processed / elapsed if elapsed else None,
        "required_unique_source_frames": sum(len(value) for value in split_indices.values()),
        "required_stream_frames": total_stream_frames,
        "projected_full_extraction_seconds_at_measured_rate": (
            total_stream_frames / (processed / elapsed) if elapsed and processed else None
        ),
        "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated(process),
        "train_only_normalization_ranges": ranges,
    }
    args.dry_run_report.parent.mkdir(parents=True, exist_ok=True)
    args.dry_run_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report


def _extract(args: argparse.Namespace, manifest: dict[str, object]) -> dict[str, object]:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite feature output: {args.output_dir}")
    sys.path.append(str(args.zarr_site_packages))
    import torch
    import transformers
    import zarr
    from PIL import Image

    group = zarr.open_group(str(args.train_zarr), mode="r")
    if group["n_channel"][:].tolist() != ["ir069", "ir107", "vil", "vis"]:
        raise ValueError("unexpected sub-SEVIR channel order")
    all_ids = [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in group["id"][:]
    ]
    train_groups = set(manifest["split_group_ids"]["train"])
    validation_groups = set(manifest["split_group_ids"]["validation"])
    if train_groups & validation_groups:
        raise ValueError("event overlap in frozen manifest")
    for window in manifest["windows"]:
        expected = train_groups if window["split"] == "train" else validation_groups
        if window["event_id"] not in expected:
            raise ValueError(f"event has inconsistent split: {window['event_id']}")
        if any(all_ids[index] != window["event_id"] for index in window["input_indices"]):
            raise ValueError(f"window source identity mismatch: {window['sample_id']}")
    split_indices = required_frame_indices(manifest["windows"])
    train_frame_set = set(split_indices["train"])
    validation_frame_set = set(split_indices["validation"])
    if train_frame_set & validation_frame_set:
        raise ValueError("source frame index overlap across splits")
    frame_indices = sorted(train_frame_set | validation_frame_set)
    frame_event_ids = [all_ids[index] for index in frame_indices]
    frame_splits = [
        "train" if index in train_frame_set else "validation"
        for index in frame_indices
    ]
    train_index_hash = hashlib.sha256(
        json.dumps(sorted(train_groups), separators=(",", ":")).encode()
    ).hexdigest()
    ranges = train_event_percentile_ranges(group, train_groups)
    processor, model = _load_model(args.model_dir, torch, transformers)
    device = torch.cuda.current_device()

    args.output_dir.mkdir(parents=True, exist_ok=False)
    indices_path = args.output_dir / "frame_indices.npy"
    splits_path = args.output_dir / "frame_splits.npy"
    ids_path = args.output_dir / "frame_event_ids.npy"
    features_path = args.output_dir / "frame_features_fp32.npy"
    np.save(indices_path, np.asarray(frame_indices, dtype=np.int64), allow_pickle=False)
    np.save(splits_path, np.asarray(frame_splits, dtype="U10"), allow_pickle=False)
    np.save(ids_path, np.asarray(frame_event_ids, dtype="U32"), allow_pickle=False)
    features = np.lib.format.open_memmap(
        features_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(frame_indices), 2, 768),
    )
    position = {source_index: offset for offset, source_index in enumerate(frame_indices)}
    batch_rows = max(1, args.batch_images // 2)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    batch_seconds: list[float] = []
    for start in range(0, len(frame_indices), batch_rows):
        indices = frame_indices[start : start + batch_rows]
        batch_start = time.perf_counter()
        encoded = _encode_rows(indices, group, ranges, processor, model, torch, Image)
        torch.cuda.synchronize(device)
        batch_seconds.append(time.perf_counter() - batch_start)
        for row_index, source_index in enumerate(indices):
            features[position[source_index]] = encoded[row_index]
        if start == 0 or (start // batch_rows + 1) % 100 == 0:
            elapsed = time.perf_counter() - started
            print(json.dumps({
                "event": "feature_extraction_progress",
                "completed_source_frames": min(start + batch_rows, len(frame_indices)),
                "total_source_frames": len(frame_indices),
                "stream_frames_per_second": (
                    2 * min(start + batch_rows, len(frame_indices)) / elapsed
                ),
                "elapsed_seconds": elapsed,
            }), flush=True)
    features.flush()
    elapsed = time.perf_counter() - started
    metadata = {
        "schema": "tiny-omni-decision/subsevir-dinov2-frame-cache-v1",
        "manifest_sha256": _file_sha256(args.manifest),
        "train_zmetadata_sha256": _file_sha256(args.train_zarr / ".zmetadata"),
        "model_id": "facebook/dinov2-small",
        "model_revision": MODEL_REVISION,
        "weights_sha256": MODEL_SHA256,
        "parameters": EXPECTED_PARAMETERS,
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "feature_definition": (
            "FP32 concat(CLS, mean(patch tokens)); each source frame is "
            "encoded independently of future frames"
        ),
        "stream_order": ["satellite", "radar"],
        "preprocessing": {
            "satellite_rgb_channels": ["ir069", "ir107", "vis"],
            "radar_rgb_channels": ["vil", "vil", "vil"],
            "train_event_ids_sha256": train_index_hash,
            "percentile_sampling": "fixed every-97th spatial pixel, p01/p99 per source channel",
            "ranges": ranges,
            "processor": processor.to_dict(),
        },
        "frame_count": len(frame_indices),
        "train_frame_count": len(train_frame_set),
        "validation_frame_count": len(validation_frame_set),
        "feature_shape": [len(frame_indices), 2, 768],
        "feature_dtype": "float32",
        "wall_seconds": elapsed,
        "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated(device),
        "mean_batch_seconds": float(np.mean(batch_seconds)) if batch_seconds else None,
        "batch_images": batch_rows * 2,
        "feature_file_sha256": _file_sha256(features_path),
        "frame_indices_sha256": _file_sha256(indices_path),
        "frame_splits_sha256": _file_sha256(splits_path),
        "frame_event_ids_sha256": _file_sha256(ids_path),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-zarr", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--zarr-site-packages", type=Path, required=True)
    parser.add_argument("--batch-images", type=int, default=32)
    parser.add_argument("--dry-run-seconds", type=int, default=0)
    parser.add_argument("--dry-run-report", type=Path, default=Path("dry-run.json"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if args.dry_run_seconds:
        _run_dry_run(args, manifest)
    else:
        if args.output_dir is None:
            parser.error("--output-dir is required unless --dry-run-seconds is set")
        result = _extract(args, manifest)
        print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
