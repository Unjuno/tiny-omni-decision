"""Read-only smoke audit for the official V-JEPA 2.1 ViT-B checkpoint.

The script loads a locally downloaded checkpoint with ``weights_only=True`` and
uses the checkpoint's pinned upstream source tree. It never trains or writes to
the model/source directories. Only the JSON audit report is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

OFFICIAL_SOURCE_COMMIT = "204698b45b3712590f06245fbfba32d3be539812"
OFFICIAL_CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def uniform_frame_indices(frame_count: int, requested_count: int) -> tuple[int, ...]:
    """Select deterministic, evenly spaced source frames, repeating if needed."""
    if frame_count < 1 or requested_count < 1:
        raise ValueError("frame_count and requested_count must be positive")
    if requested_count == 1:
        return (0,)
    if frame_count == 1:
        return (0,) * requested_count
    return tuple(
        round(index * (frame_count - 1) / (requested_count - 1)) for index in range(requested_count)
    )


def expected_patch_tokens(frames: int, resolution: int, patch_size: int, tubelet: int) -> int:
    if min(frames, resolution, patch_size, tubelet) < 1:
        raise ValueError("frame and patch dimensions must be positive")
    if frames % tubelet or resolution % patch_size:
        raise ValueError("input dimensions must be divisible by patch dimensions")
    return (frames // tubelet) * (resolution // patch_size) ** 2


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--video-frames", type=int, default=8)
    return parser.parse_args()


def _git_revision(source_root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _decode_video(path: Path, requested_count: int) -> tuple[list[Any], tuple[int, ...], int]:
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(stream)]
    if not frames:
        raise ValueError(f"video contains no decodable frames: {path}")
    selected = uniform_frame_indices(len(frames), requested_count)
    return [frames[index] for index in selected], selected, len(frames)


def _clean_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    # This mirrors the pinned official source helper without mutating the
    # mmap-backed checkpoint dictionaries used for the independent load.
    cleaned = {}
    for key, value in state_dict.items():
        cleaned[key.replace("module.", "").replace("backbone.", "")] = value
    return cleaned


def _run_encoder(encoder: Any, inputs: Any, torch: Any) -> tuple[Any, float]:
    start = time.perf_counter()
    with torch.inference_mode():
        output = encoder(inputs)
    if not torch.isfinite(output).all():
        raise ValueError("encoder output contains NaN or Inf")
    return output, time.perf_counter() - start


def main() -> None:
    args = _parse_args()
    source_root = args.source_root.resolve()
    checkpoint_path = args.checkpoint.resolve()
    image_path = args.image.resolve()
    video_path = args.video.resolve()
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite audit report: {output_path}")
    if not all(path.is_file() for path in (checkpoint_path, image_path, video_path)):
        raise FileNotFoundError("checkpoint, image, and video must all be existing files")
    source_revision = _git_revision(source_root)
    if source_revision != OFFICIAL_SOURCE_COMMIT:
        raise ValueError(f"unexpected upstream source revision: {source_revision}")
    checkpoint_hash = sha256_file(checkpoint_path)
    if checkpoint_hash.lower() != args.expected_checkpoint_sha256.lower():
        raise ValueError("checkpoint SHA-256 did not match the requested pin")
    if args.video_frames < 2:
        raise ValueError("a video smoke requires at least two requested frames")

    import av
    import numpy as np
    import torch
    from PIL import Image

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    torch.set_num_threads(4)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    import sys

    sys.path.insert(0, str(source_root))
    from app.vjepa_2_1.models import vision_transformer
    from evals.hub.preprocessor import vjepa2_preprocessor
    from src.hub import backbones

    started = time.perf_counter()
    # mmap avoids copying the 1.66 GB training checkpoint into an extra buffer.
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    required_keys = {"ema_encoder", "encoder", "predictor"}
    if not required_keys.issubset(checkpoint):
        missing_keys = sorted(required_keys - set(checkpoint))
        raise ValueError(f"checkpoint is missing required model keys: {missing_keys}")
    checkpoint_keys = sorted(checkpoint)
    checkpoint_epoch = checkpoint.get("epoch")
    ema_state = checkpoint["ema_encoder"]
    predictor_state = checkpoint["predictor"]
    ema_tensor_count = len(ema_state)
    ema_elements = sum(value.numel() for value in ema_state.values())
    ema_bytes = sum(value.numel() * value.element_size() for value in ema_state.values())
    ema_dtypes = sorted({str(value.dtype) for value in ema_state.values()})

    # Exercise the pinned upstream factory and strict checkpoint loading. The
    # upstream helper currently points at localhost:8300; inject the already
    # safely loaded official checkpoint instead of changing upstream files.
    expected_model_url = "http://localhost:8300/vjepa2_1_vitb_dist_vitG_384.pt"
    original_loader = torch.hub.load_state_dict_from_url

    def local_checkpoint_loader(
        url: str, *loader_args: Any, **loader_kwargs: Any
    ) -> dict[str, Any]:
        if url != expected_model_url:
            raise ValueError(f"unexpected URL requested by pinned factory: {url}")
        return {"ema_encoder": dict(ema_state), "predictor": dict(predictor_state)}

    torch.hub.load_state_dict_from_url = local_checkpoint_loader
    try:
        encoder, training_predictor = backbones.vjepa2_1_vit_base_384(pretrained=True)
    finally:
        torch.hub.load_state_dict_from_url = original_loader
    del training_predictor

    # Independently load the exact EMA state into the source ViT-B module and
    # compare its numerical output with the official factory-loaded encoder.
    reference_encoder = vision_transformer.vit_base(
        patch_size=16,
        img_size=(384, 384),
        num_frames=64,
        tubelet_size=2,
        use_sdpa=True,
        use_silu=False,
        wide_silu=True,
        uniform_power=False,
        use_rope=True,
        img_temporal_dim_size=1,
        interpolate_rope=True,
        n_output_distillation=1,
    )
    reference_encoder.load_state_dict(_clean_state_dict(ema_state), strict=True)
    checkpoint = None
    ema_state = None
    predictor_state = None

    transform = vjepa2_preprocessor(pretrained=False, crop_size=384)
    with Image.open(image_path) as image:
        image_frames = [np.asarray(image.convert("RGB"))]
    image_input = transform(image_frames)[0].unsqueeze(0).to(device=device, dtype=torch.float32)
    video_frames, selected_indices, decoded_count = _decode_video(video_path, args.video_frames)
    video_input = transform(video_frames)[0].unsqueeze(0).to(device=device, dtype=torch.float32)

    encoder = encoder.eval().to(device)
    reference_encoder = reference_encoder.eval().to(device)
    image_output, image_seconds = _run_encoder(encoder, image_input, torch)
    image_reference, image_reference_seconds = _run_encoder(reference_encoder, image_input, torch)
    video_output, video_seconds = _run_encoder(encoder, video_input, torch)
    video_reference, video_reference_seconds = _run_encoder(reference_encoder, video_input, torch)

    image_max_abs_diff = float((image_output - image_reference).abs().max().item())
    video_max_abs_diff = float((video_output - video_reference).abs().max().item())
    if not torch.allclose(image_output, image_reference, atol=1e-5, rtol=1e-5):
        raise ValueError("official factory and direct EMA image outputs diverged")
    if not torch.allclose(video_output, video_reference, atol=1e-5, rtol=1e-5):
        raise ValueError("official factory and direct EMA video outputs diverged")

    image_expected_tokens = expected_patch_tokens(1, 384, 16, 1)
    video_expected_tokens = expected_patch_tokens(args.video_frames, 384, 16, 2)
    if tuple(image_output.shape) != (1, image_expected_tokens, 768):
        raise ValueError(f"unexpected image token shape: {tuple(image_output.shape)}")
    if tuple(video_output.shape) != (1, video_expected_tokens, 768):
        raise ValueError(f"unexpected video token shape: {tuple(video_output.shape)}")

    named_parameters = tuple(encoder.named_parameters())
    peak_allocated = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    peak_reserved = torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
    report = {
        "schema_version": 1,
        "status": "complete_read_only_checkpoint_and_forward_audit",
        "source": {
            "repository": "https://github.com/facebookresearch/vjepa2",
            "revision": source_revision,
            "license": (
                "MIT repository license; checkpoint-specific license statement not found "
                "in release README"
            ),
            "checkpoint_url": OFFICIAL_CHECKPOINT_URL,
            "checkpoint_license_status": "REVIEW_REQUIRED",
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "bytes": checkpoint_path.stat().st_size,
            "sha256": checkpoint_hash,
            "load_mode": "torch.load(weights_only=True, mmap=True)",
            "top_level_keys": checkpoint_keys,
            "epoch": checkpoint_epoch,
            "ema_encoder_tensor_count": ema_tensor_count,
            "ema_encoder_elements": ema_elements,
            "ema_encoder_bytes": ema_bytes,
            "ema_encoder_dtypes": ema_dtypes,
            "predictor_loaded_by_official_factory": True,
            "strict_state_dict_load": True,
        },
        "model": {
            "architecture": "V-JEPA 2.1 ViT-B/16",
            "encoder_parameters": sum(parameter.numel() for _, parameter in named_parameters),
            "deployable_encoder_parameters_only": True,
            "predictor_excluded_from_runtime_count": True,
            "configured_image_size": [384, 384],
            "patch_size": 16,
            "tubelet_size": 2,
            "loaded_dtype": "float32",
            "device": str(device),
        },
        "preprocessing": {
            "implementation": "pinned upstream evals.hub.preprocessor.vjepa2_preprocessor",
            "source_revision": source_revision,
            "crop_size": 384,
            "normalization_mean": [0.485, 0.456, 0.406],
            "normalization_std": [0.229, 0.224, 0.225],
            "image_path": str(image_path),
            "image_sha256": sha256_file(image_path),
            "video_path": str(video_path),
            "video_sha256": sha256_file(video_path),
            "requested_video_frames": args.video_frames,
            "decoded_video_frames": decoded_count,
            "selected_frame_indices": list(selected_indices),
        },
        "forward": {
            "image_output_shape": list(image_output.shape),
            "image_tokens": image_expected_tokens,
            "video_output_shape": list(video_output.shape),
            "video_tokens": video_expected_tokens,
            "hidden_size": 768,
            "finite_outputs": True,
            "official_loader_vs_direct_ema_image_max_abs_diff": image_max_abs_diff,
            "official_loader_vs_direct_ema_video_max_abs_diff": video_max_abs_diff,
            "image_seconds": image_seconds,
            "image_reference_seconds": image_reference_seconds,
            "video_seconds": video_seconds,
            "video_reference_seconds": video_reference_seconds,
            "cuda_peak_allocated_bytes": peak_allocated,
            "cuda_peak_reserved_bytes": peak_reserved,
        },
        "environment": {
            "python": __import__("sys").version.split()[0],
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "timm": __import__("timm").__version__,
            "einops": __import__("einops").__version__,
            "av": av.__version__,
            "elapsed_seconds": time.perf_counter() - started,
        },
        "limitations": [
            (
                "The local smoke image/video files are plumbing fixtures, not quality "
                "evaluation samples."
            ),
            (
                "The official source repository is MIT licensed, but the downloaded "
                "checkpoint has no explicit separate license in its release documentation; "
                "product use remains gated on rights review."
            ),
            (
                "This is an image/video encoder only; it does not demonstrate text/audio "
                "fusion or Decision quality."
            ),
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
