"""Extract pinned frozen VideoMAE image features from a Clevr-4 ZIP manifest."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import sys
import time
import zipfile
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from scripts.audit_clevr4_content_overlap import _load_jsonl, sha256_file

MODEL_DIR = Path("C:/CodexArtifacts/pretrained-reuse-r1/videomaev2-small-audit-01")
SOURCE_SHA256 = "d132c97369f25207e380e78c59294fd00b93d2f6aa0fb5c4ded19eff26a5a3cb"
CHECKPOINT_SHA256 = "24fb71687fa3671b8387cadfbcbab0f72af695692e93cf1ecc82caa888626172"
PARAMETER_COUNT = 22_153_286
FEATURE_WIDTH = 384
FRAME_COUNT = 16
IMAGE_SIZE = 224
ARCHIVE_BYTES = 36_057_238_315


def image_tensor(image_bytes: bytes) -> torch.Tensor:
    """Return static RGB image repeated over VideoMAE's fixed 16-frame input."""
    with Image.open(BytesIO(image_bytes)) as image:
        pixels = np.asarray(
            image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS),
            dtype=np.uint8,
        ).copy()
    frame = torch.from_numpy(pixels).permute(2, 0, 1).float().div_(255.0)
    frame.sub_(0.5).div_(0.5)
    return frame.unsqueeze(1).expand(3, FRAME_COUNT, IMAGE_SIZE, IMAGE_SIZE).contiguous()


def load_encoder() -> torch.nn.Module:
    source = MODEL_DIR / "modeling_finetune.py"
    checkpoint = MODEL_DIR / "vit_s_k710_dl_from_giant.pth"
    if sha256_file(source) != SOURCE_SHA256 or sha256_file(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("VideoMAE source/checkpoint hash differs from the pinned runtime audit")
    sys.path.insert(0, "C:/CodexArtifacts/pretrained-reuse-r1/python-deps")
    spec = importlib.util.spec_from_file_location("videomae_clevr4_source", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not import the pinned VideoMAE source")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)["module"]
    model = module.vit_small_patch16_224(num_classes=710).eval().requires_grad_(False)
    model.load_state_dict(state, strict=True)
    count = sum(parameter.numel() for parameter in model.parameters())
    if count != PARAMETER_COUNT:
        raise ValueError(f"unexpected VideoMAE parameter count: {count}")
    return model


def extract(
    *,
    archive: Path,
    manifest: Path,
    output: Path,
    batch_size: int = 8,
    max_images: int | None = None,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output {output}")
    if archive.stat().st_size != ARCHIVE_BYTES:
        raise ValueError("archive size differs from the pinned Clevr-4 preflight")
    if batch_size <= 0 or (max_images is not None and max_images <= 0):
        raise ValueError("batch_size and max_images must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("this frozen-feature run requires the local CUDA GPU")
    rows = _load_jsonl(manifest)
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("empty media manifest or duplicate IDs")
    if max_images is not None:
        rows = rows[:max_images]
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda")
    torch.manual_seed(17)
    torch.cuda.reset_peak_memory_stats(device)
    model_load_started = time.perf_counter()
    model = load_encoder().to(device)
    model_load_seconds = time.perf_counter() - model_load_started
    started_at = datetime.now(UTC).isoformat()
    started = time.perf_counter()
    features: list[np.ndarray] = []
    with zipfile.ZipFile(archive) as source:
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start : start + batch_size]
            tensors = []
            for row in batch_rows:
                image_bytes = source.read(row["archive_member"])
                if hashlib.sha256(image_bytes).hexdigest() != row["media_sha256"]:
                    raise ValueError(f"media SHA-256 mismatch for {row['id']}")
                tensors.append(image_tensor(image_bytes))
            inputs = torch.stack(tensors).to(device)
            with torch.inference_mode():
                encoded = model.forward_features(inputs)
            if encoded.shape != (len(batch_rows), FEATURE_WIDTH):
                raise ValueError(f"unexpected feature shape {tuple(encoded.shape)}")
            if not torch.isfinite(encoded).all():
                raise ValueError("non-finite encoder output")
            features.append(encoded.float().cpu().numpy())
            if (start + len(batch_rows)) % max(batch_size * 16, 128) == 0:
                print(f"encoded {start + len(batch_rows)}/{len(rows)} images", flush=True)
    matrix = np.concatenate(features, axis=0).astype(np.float32, copy=False)
    np.savez_compressed(
        output / "frozen-features.npz",
        ids=np.asarray([row["id"] for row in rows], dtype=np.str_),
        media_sha256=np.asarray([row["media_sha256"] for row in rows], dtype=np.str_),
        features=matrix,
    )
    (output / "media-rows.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    result = {
        "status": "COMPLETE_FROZEN_STATIC_IMAGE_FEATURE_EXTRACTION",
        "started_at_utc": started_at,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "archive_bytes": archive.stat().st_size,
        "archive_bytes_expected": ARCHIVE_BYTES,
        "archive_sha512_recomputed": False,
        "input_manifest_sha256": sha256_file(manifest),
        "input_row_count": len(rows),
        "feature_shape": list(matrix.shape),
        "feature_dtype": str(matrix.dtype),
        "feature_sha256": sha256_file(output / "frozen-features.npz"),
        "media_rows_sha256": sha256_file(output / "media-rows.jsonl"),
        "encoder": "OpenGVLab VideoMAE V2 ViT-Small K710",
        "encoder_source_sha256": SOURCE_SHA256,
        "encoder_checkpoint_sha256": CHECKPOINT_SHA256,
        "encoder_parameters": PARAMETER_COUNT,
        "frozen": True,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "dtype": "float32",
        "batch_size": batch_size,
        "frames": FRAME_COUNT,
        "image_mode": "one static image repeated across all frames; no temporal evidence",
        "preprocessing": "RGB Lanczos resize 224; /255; mean/std 0.5",
        "torch": torch.__version__,
        "python": platform.python_version(),
        "wall_seconds": time.perf_counter() - started,
        "model_load_seconds": model_load_seconds,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "source_files_modified": False,
        "sealed_audit_loaded": False,
    }
    (output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-images", type=int)
    args = parser.parse_args()
    extract(
        archive=args.archive,
        manifest=args.manifest,
        output=args.output,
        batch_size=args.batch_size,
        max_images=args.max_images,
    )


if __name__ == "__main__":
    main()
