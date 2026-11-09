"""CPU-only VideoMAEv2 Path A encoder serialization/reload smoke; no quality test."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

MODEL_ROOT = Path("C:/CodexArtifacts/pretrained-reuse-r1/videomaev2-small-audit-01")
DEPENDENCY_ROOT = Path("C:/CodexArtifacts/pretrained-reuse-r1/python-deps")
CHECKPOINT_SHA256 = "24fb71687fa3671b8387cadfbcbab0f72af695692e93cf1ecc82caa888626172"
MODEL_SOURCE_SHA256 = "d132c97369f25207e380e78c59294fd00b93d2f6aa0fb5c4ded19eff26a5a3cb"
IMAGE_SIZE = 224
FRAME_COUNT = 16


def normalize_rgb_frames(frames: np.ndarray) -> np.ndarray:
    """Normalize uint8 RGB frames into VideoMAEv2 [1,C,T,H,W] CPU input."""
    if frames.shape != (FRAME_COUNT, IMAGE_SIZE, IMAGE_SIZE, 3):
        raise ValueError("frames must have shape [16,224,224,3]")
    if frames.dtype != np.uint8:
        raise ValueError("frames must be uint8 RGB")
    values = frames.astype(np.float32) / 255.0
    values = (values - 0.5) / 0.5
    return np.transpose(values, (3, 0, 1, 2))[None, ...].copy()


def static_image_clip(image: np.ndarray) -> np.ndarray:
    """Represent one image as 16 identical frames for a frozen video encoder."""
    if image.shape != (IMAGE_SIZE, IMAGE_SIZE, 3) or image.dtype != np.uint8:
        raise ValueError("image must be a uint8 RGB array with shape [224,224,3]")
    return np.repeat(image[None, ...], FRAME_COUNT, axis=0)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _load_model() -> tuple[Any, Any, Any]:
    import torch

    checkpoint_path = MODEL_ROOT / "vit_s_k710_dl_from_giant.pth"
    model_source_path = MODEL_ROOT / "modeling_finetune.py"
    if _sha256_bytes(checkpoint_path.read_bytes()) != CHECKPOINT_SHA256:
        raise ValueError("VideoMAEv2 checkpoint hash mismatch")
    if _sha256_bytes(model_source_path.read_bytes()) != MODEL_SOURCE_SHA256:
        raise ValueError("VideoMAEv2 model source hash mismatch")

    sys.path.insert(0, str(DEPENDENCY_ROOT))
    spec = importlib.util.spec_from_file_location("path_a_videomaev2_source", model_source_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load pinned VideoMAEv2 source")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    torch.set_num_threads(4)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)["module"]
    model = module.vit_small_patch16_224(num_classes=710).cpu().eval().requires_grad_(False)
    model.load_state_dict(state, strict=True)
    return torch, module, model


def run_roundtrip(output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    started = time.perf_counter()
    torch, module, model = _load_model()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != 22_153_286:
        raise ValueError(f"unexpected loaded parameter count: {parameter_count}")

    rng = np.random.default_rng(17)
    image = rng.integers(0, 256, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    video = rng.integers(
        0,
        256,
        (FRAME_COUNT, IMAGE_SIZE, IMAGE_SIZE, 3),
        dtype=np.uint8,
    )
    inputs = {
        "image_repeated_16_frames": normalize_rgb_frames(static_image_clip(image)),
        "synthetic_16_frame_video": normalize_rgb_frames(video),
    }

    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    serialized_state = stream.getvalue()
    stream.seek(0)
    reloaded_model = module.vit_small_patch16_224(num_classes=710)
    reloaded_model = reloaded_model.cpu().eval().requires_grad_(False)
    loaded_state = torch.load(stream, map_location="cpu", weights_only=True)
    reloaded_model.load_state_dict(loaded_state, strict=True)

    route_results: dict[str, dict[str, Any]] = {}
    with torch.inference_mode():
        for route, array in inputs.items():
            batch = torch.from_numpy(array)
            expected_logits = model(batch)
            expected_features = model.forward_features(batch)
            actual_logits = reloaded_model(batch)
            actual_features = reloaded_model.forward_features(batch)
            finite = bool(
                torch.isfinite(expected_logits).all()
                and torch.isfinite(expected_features).all()
            )
            exact = bool(
                torch.equal(expected_logits, actual_logits)
                and torch.equal(expected_features, actual_features)
            )
            if not finite or not exact:
                raise ValueError(f"non-finite output or reload mismatch for {route}")
            route_results[route] = {
                "logits_shape": list(expected_logits.shape),
                "features_shape": list(expected_features.shape),
                "finite": finite,
                "bitwise_equal_after_serialization_reload": exact,
                "input_sha256": _sha256_bytes(array.tobytes()),
            }

    output.mkdir(parents=True, exist_ok=False)
    result = {
        "status": "PASS_CPU_STRICT_LOAD_AND_SERIALIZATION_RELOAD_SMOKE_NOT_QUALITY",
        "source_commit": "a15a0becc1007a1af56db17ba9f2ea9fb06d286c",
        "device": "cpu",
        "gpu_used": False,
        "model_id": "OpenGVLab/VideoMAE2-vit-small-k710-distilled",
        "model_source_sha256": MODEL_SOURCE_SHA256,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "checkpoint_bytes": (MODEL_ROOT / "vit_s_k710_dl_from_giant.pth").stat().st_size,
        "serialized_state_sha256": _sha256_bytes(serialized_state),
        "serialized_state_bytes": len(serialized_state),
        "parameters": parameter_count,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "av": importlib.metadata.version("av"),
        "timm": importlib.metadata.version("timm"),
        "frame_count": FRAME_COUNT,
        "input_shape": [1, 3, FRAME_COUNT, IMAGE_SIZE, IMAGE_SIZE],
        "preprocessing": "synthetic RGB uint8; /255 then mean/std 0.5; image repeated 16x",
        "routes": route_results,
        "wall_seconds": time.perf_counter() - started,
        "claims": {
            "dataset_loaded": False,
            "validation_scored": False,
            "training_performed": False,
            "sealed_audit_opened": False,
            "path_a_dispatcher_integrated": False,
            "decision_quality_measured": False,
        },
    }
    (output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_roundtrip(args.output.resolve()), sort_keys=True))


if __name__ == "__main__":
    main()
