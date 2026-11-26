"""GPU smoke for frozen DINOv2 frame features on one train-only sub-SEVIR window."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from tiny_omni_decision.subsevir_features import (
    cls_plus_patch_mean,
    frame_to_rgb,
    train_event_percentile_ranges,
)

MODEL_REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
MODEL_SHA256 = "ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1"
EXPECTED_PARAMETERS = 22_056_576


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decode(value: object) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite DINOv2 smoke output: {args.output}")
    sys.path.append(str(args.zarr_site_packages))
    import torch
    import zarr
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    train_groups = set(manifest["split_group_ids"]["train"])
    validation_groups = set(manifest["split_group_ids"]["validation"])
    if train_groups & validation_groups:
        raise ValueError("event overlap in manifest")
    window = next(row for row in manifest["windows"] if row["split"] == "train")
    if window["event_id"] not in train_groups:
        raise ValueError("selected smoke window is not a training event")

    group = zarr.open_group(str(args.train_zarr), mode="r")
    if group["n_channel"][:].tolist() != ["ir069", "ir107", "vil", "vis"]:
        raise ValueError("unexpected source channel order")
    ranges = train_event_percentile_ranges(group, train_groups)
    ids = [_decode(value) for value in group["id"][:]]
    if any(ids[index] not in train_groups for index in window["input_indices"]):
        raise ValueError("smoke input rows are not in a training event")
    raw = np.asarray(group["features"][list(window["input_indices"])], dtype=np.float32)

    images = []
    mapping = []
    for stream in ("satellite", "radar"):
        for frame_position in range(raw.shape[0]):
            rgb = frame_to_rgb(raw[frame_position], ranges, stream)
            images.append(Image.fromarray(rgb, mode="RGB"))
            mapping.append({"stream": stream, "frame_position": frame_position})

    weight_file = args.model_dir / "model.safetensors"
    if sha256_file(weight_file) != MODEL_SHA256:
        raise ValueError("DINOv2 local weight hash mismatch")
    processor = AutoImageProcessor.from_pretrained(args.model_dir, local_files_only=True)
    model, loading = AutoModel.from_pretrained(
        args.model_dir,
        local_files_only=True,
        output_loading_info=True,
    )
    load_error_keys = ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    if any(loading.get(key) for key in load_error_keys):
        raise ValueError(f"non-strict DINOv2 load: {loading}")
    if sum(parameter.numel() for parameter in model.parameters()) != EXPECTED_PARAMETERS:
        raise ValueError("unexpected DINOv2 parameter count")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for the requested local GPU smoke")
    device = torch.device("cuda")
    model.to(device).eval().requires_grad_(False)
    encoded = processor(images=images, return_tensors="pt")
    pixel_values = encoded["pixel_values"].to(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with torch.inference_mode():
        output = model(pixel_values=pixel_values).last_hidden_state
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    features_cpu = cls_plus_patch_mean(output.float().cpu().numpy())
    if features_cpu.shape != (16, 768) or not np.isfinite(features_cpu).all():
        raise ValueError(f"invalid DINOv2 feature shape/values: {features_cpu.shape}")

    per_stream = {}
    for stream in ("satellite", "radar"):
        stream_features = np.stack(
            [
                features_cpu[index]
                for index, item in enumerate(mapping)
                if item["stream"] == stream
            ]
        )
        per_stream[stream] = {
            "sequence_shape": list(stream_features.shape),
            "feature_sha256": hashlib.sha256(stream_features.astype("<f4").tobytes()).hexdigest(),
            "finite": bool(np.isfinite(stream_features).all()),
        }
    result = {
        "schema": "tiny-omni-decision/subsevir-dinov2-smoke-v1",
        "status": "pass",
        "scope": "one train-only event, frozen image encoder; no fitting or validation access",
        "manifest_sha256": sha256_file(args.manifest),
        "train_zmetadata_sha256": sha256_file(args.train_zarr / ".zmetadata"),
        "model_id": "facebook/dinov2-small",
        "model_revision": MODEL_REVISION,
        "weights_sha256": MODEL_SHA256,
        "parameters": EXPECTED_PARAMETERS,
        "load_info": loading,
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "dtype": str(next(model.parameters()).dtype),
        "sample_id": window["sample_id"],
        "normalization": {
            "method": "train-event-only every-97th-pixel sample, per-channel p01/p99 to uint8",
            "ranges": ranges,
            "satellite_rgb_channels": ["ir069", "ir107", "vis"],
            "radar_rgb_channels": ["vil", "vil", "vil"],
            "processor": processor.to_dict(),
        },
        "input_shape": list(pixel_values.shape),
        "per_stream": per_stream,
        "forward_seconds": elapsed,
        "examples_per_second_for_16_frames": 16 / elapsed if elapsed else None,
        "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated(device),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-zarr", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--zarr-site-packages", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    result = run(parser.parse_args())
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
