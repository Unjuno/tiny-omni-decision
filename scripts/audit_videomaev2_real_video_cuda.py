"""Run a pinned VideoMAE V2 checkpoint on one hash-verified training video.

This is an R1 runtime feasibility audit, not a task-quality evaluation.
The output directory must be new. Existing model, media, and audit artifacts
are read-only.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import av
import numpy as np
import timm
import torch

SOURCE_SHA256 = "d132c97369f25207e380e78c59294fd00b93d2f6aa0fb5c4ded19eff26a5a3cb"
CHECKPOINT_SHA256 = "24fb71687fa3671b8387cadfbcbab0f72af695692e93cf1ecc82caa888626172"
VIDEO_SHA256 = "6e61d8f411114ef7f052a5bea40bd51b4161274bb0d799cd158658a14b6d4df7"
EXPECTED_PARAMETERS = 22_153_286


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model-root",
        type=Path,
        default=Path("C:/CodexArtifacts/pretrained-reuse-r1/videomaev2-small-audit-01"),
    )
    parser.add_argument(
        "--dependency-root",
        type=Path,
        default=Path("C:/CodexArtifacts/pretrained-reuse-r1/python-deps"),
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=Path(
            "C:/CodexArtifacts/pretrained-reuse-r2/"
            "perception-occlusion-train-media-01/video_10186.mp4"
        ),
    )
    args = parser.parse_args()

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = args.model_root / "modeling_finetune.py"
    checkpoint = args.model_root / "vit_s_k710_dl_from_giant.pth"
    for path, expected, label in (
        (source, SOURCE_SHA256, "VideoMAE V2 source"),
        (checkpoint, CHECKPOINT_SHA256, "VideoMAE V2 checkpoint"),
        (args.video, VIDEO_SHA256, "training video"),
    ):
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"{label} SHA-256 mismatch: {actual}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; this audit requires the local GPU")

    sys.path.insert(0, str(args.dependency_root))
    spec = importlib.util.spec_from_file_location("videomae_audit_source", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load pinned VideoMAE source")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    torch.set_num_threads(4)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)["module"]
    model = module.vit_small_patch16_224(num_classes=710).eval().requires_grad_(False)
    model.load_state_dict(state, strict=True)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != EXPECTED_PARAMETERS:
        raise ValueError(f"unexpected parameter count: {parameter_count}")
    model = model.to("cuda")

    selected_frames = np.linspace(0, 650, 16).astype(int).tolist()
    decoded: dict[int, np.ndarray] = {}
    with av.open(str(args.video)) as container:
        for frame_index, frame in enumerate(container.decode(video=0)):
            if frame_index in selected_frames:
                decoded[frame_index] = frame.reformat(
                    width=224, height=224, format="rgb24"
                ).to_ndarray()
            if frame_index >= 650:
                break
    if sorted(decoded) != selected_frames:
        raise ValueError("training video did not decode all predeclared frame indices")

    pixels = np.stack([decoded[index] for index in selected_frames])
    inputs = torch.from_numpy(pixels.copy()).permute(3, 0, 1, 2).float()[None] / 255
    inputs = ((inputs - 0.5) / 0.5).to("cuda")

    with torch.inference_mode():
        _ = model(inputs)
        torch.cuda.synchronize()
        baseline_allocated = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        logits = model(inputs)
        features = model.forward_features(inputs)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        repeated_logits = model(inputs)
        torch.cuda.synchronize()

    if not torch.isfinite(logits).all() or not torch.isfinite(features).all():
        raise FloatingPointError("non-finite VideoMAE output")
    torch.testing.assert_close(logits, repeated_logits, rtol=0, atol=0)

    result = {
        "status": "PASS_CUDA_REAL_TRAIN_VIDEO_FORWARD_NOT_QUALITY",
        "source_revision": "OpenGVLab/VideoMAE2@706cc172d65ebd4dedbee3f9c0183a93df9fa125",
        "source_commit": "29eab1e8a588d1b3ec0cdec7b03a86cca491b74b",
        "source_sha256": SOURCE_SHA256,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "checkpoint_bytes": checkpoint.stat().st_size,
        "model_card_license": "Apache-2.0 (pinned owner card)",
        "parameters": parameter_count,
        "video_id": "video_10186",
        "video_split": "train",
        "video_sha256": VIDEO_SHA256,
        "frame_indices": selected_frames,
        "preprocessing": (
            "diagnostic RGB resize 224; /255 then mean/std 0.5; not official multi-crop evaluation"
        ),
        "logits_shape": list(logits.shape),
        "features_shape": list(features.shape),
        "finite_outputs": True,
        "repeat_logits_bitwise_equal": True,
        "forward_seconds_logits_and_features": elapsed,
        "device": torch.cuda.get_device_name(0),
        "device_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "av": av.__version__,
        "timm": timm.__version__,
        "dtype": str(inputs.dtype),
        "allocated_bytes_with_model_and_input": baseline_allocated,
        "peak_allocated_bytes_with_model_and_input": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes_with_model_and_input": torch.cuda.max_memory_reserved(),
        "quality_evaluation": False,
        "validation_or_audit_read": False,
        "training": False,
    }
    with (output / "runtime.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    np.save(output / "features.npy", features.cpu().numpy(), allow_pickle=False)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
