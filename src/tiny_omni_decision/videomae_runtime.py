"""Pinned VideoMAE V2 Clevr-4 image Decision runtime."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn

from tiny_omni_decision.dataset import CLEVR4_TAXONOMIES

ENCODER_SOURCE_SHA256 = "d132c97369f25207e380e78c59294fd00b93d2f6aa0fb5c4ded19eff26a5a3cb"
ENCODER_CHECKPOINT_SHA256 = "24fb71687fa3671b8387cadfbcbab0f72af695692e93cf1ecc82caa888626172"
READOUT_SHA256 = "437e920e7cda02ff2e28f19c96433af04aab3d70494030b6e3278809aa1cc39e"
ENCODER_PARAMETERS = 22_153_286
READOUT_PARAMETERS = 52_682
IMAGE_SIZE = 224
FRAME_COUNT = 16
FEATURE_WIDTH = 384
HIDDEN_WIDTH = 128
TASK_EMBED_WIDTH = 16
TASK_NAMES = list(CLEVR4_TAXONOMIES)
TASK_TO_INDEX = {name: index for index, name in enumerate(TASK_NAMES)}
OPTION_TO_INDEX = {
    task: {str(option): index for index, option in enumerate(options)}
    for task, options in CLEVR4_TAXONOMIES.items()
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_tensor(image_bytes: bytes) -> Tensor:
    """Apply the frozen Path A image preprocessing and repeat across 16 frames."""
    with Image.open(io.BytesIO(image_bytes)) as image:
        pixels = np.asarray(
            image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS),
            dtype=np.uint8,
        ).copy()
    frame = torch.from_numpy(pixels).permute(2, 0, 1).float().div_(255.0)
    frame.sub_(0.5).div_(0.5)
    return frame.unsqueeze(1).expand(3, FRAME_COUNT, IMAGE_SIZE, IMAGE_SIZE).contiguous()


class TaxonomyConditionedScorer(nn.Module):
    """Same small scorer architecture used by the pinned Path A training run."""

    def __init__(self) -> None:
        super().__init__()
        self.task_embedding = nn.Embedding(len(TASK_NAMES), TASK_EMBED_WIDTH)
        self.network = nn.Sequential(
            nn.Linear(FEATURE_WIDTH + TASK_EMBED_WIDTH, HIDDEN_WIDTH),
            nn.GELU(),
            nn.Linear(HIDDEN_WIDTH, max(len(values) for values in CLEVR4_TAXONOMIES.values())),
        )

    def forward(self, image_features: Tensor, task_ids: Tensor) -> Tensor:
        return self.network(torch.cat([image_features, self.task_embedding(task_ids)], dim=-1))


def _load_encoder(model_root: Path, dependency_root: Path) -> torch.nn.Module:
    source = model_root / "modeling_finetune.py"
    checkpoint = model_root / "vit_s_k710_dl_from_giant.pth"
    if sha256_file(source) != ENCODER_SOURCE_SHA256:
        raise ValueError("VideoMAE source hash mismatch")
    if sha256_file(checkpoint) != ENCODER_CHECKPOINT_SHA256:
        raise ValueError("VideoMAE checkpoint hash mismatch")
    sys.path.insert(0, str(dependency_root.resolve(strict=True)))
    spec = importlib.util.spec_from_file_location("path_a_videomae_runtime_source", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not import the pinned VideoMAE source")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)["module"]
    encoder = module.vit_small_patch16_224(num_classes=710).eval().requires_grad_(False)
    encoder.load_state_dict(state, strict=True)
    count = sum(parameter.numel() for parameter in encoder.parameters())
    if count != ENCODER_PARAMETERS:
        raise ValueError(f"unexpected VideoMAE parameter count: {count}")
    return encoder


class VideoMAEClevr4ImageRuntime:
    """Score supplied Clevr-4 taxonomy options from one local image."""

    def __init__(
        self,
        model_root: Path,
        readout_path: Path,
        dependency_root: Path,
        *,
        device: str,
        encoder: torch.nn.Module | None = None,
    ) -> None:
        if device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("VideoMAE CUDA runtime requested but CUDA is unavailable")
        self.device = torch.device(device)
        torch.set_num_threads(4)
        self.encoder = (
            encoder
            if encoder is not None
            else _load_encoder(model_root.resolve(strict=True), dependency_root)
        )
        self.encoder.to(self.device).eval().requires_grad_(False)
        if sha256_file(readout_path) != READOUT_SHA256:
            raise ValueError("VideoMAE Clevr-4 readout hash mismatch")
        self.scorer = TaxonomyConditionedScorer().to(self.device).eval().requires_grad_(False)
        state = torch.load(readout_path, map_location=self.device, weights_only=True)
        self.scorer.load_state_dict(state, strict=True)
        if sum(parameter.numel() for parameter in self.scorer.parameters()) != READOUT_PARAMETERS:
            raise ValueError("unexpected VideoMAE readout parameter count")

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "encoder": "OpenGVLab VideoMAE V2 ViT-Small K710",
            "encoder_source_sha256": ENCODER_SOURCE_SHA256,
            "encoder_checkpoint_sha256": ENCODER_CHECKPOINT_SHA256,
            "readout_sha256": READOUT_SHA256,
            "encoder_parameters": ENCODER_PARAMETERS,
            "readout_parameters": READOUT_PARAMETERS,
            "unique_parameters": ENCODER_PARAMETERS + READOUT_PARAMETERS,
            "device": torch.cuda.get_device_name(self.device)
            if self.device.type == "cuda"
            else "cpu",
            "dtype": str(next(self.encoder.parameters()).dtype),
            "image_preprocessing": (
                "RGB Lanczos 224; /255; mean/std 0.5; repeat static image across 16 frames"
            ),
            "temporal_evidence": False,
        }

    def decide(
        self,
        image_path: Path,
        taxonomy: str,
        options: list[str],
        *,
        expected_media_sha256: str,
    ) -> dict[str, Any]:
        if taxonomy not in OPTION_TO_INDEX:
            raise ValueError(f"unknown Clevr-4 taxonomy: {taxonomy!r}")
        if len(options) < 2 or len(set(options)) != len(options):
            raise ValueError("at least two unique supplied options are required")
        try:
            option_indices = [OPTION_TO_INDEX[taxonomy][str(option)] for option in options]
        except KeyError as error:
            raise ValueError(
                f"option {error.args[0]!r} is not valid for taxonomy {taxonomy}"
            ) from error
        path = image_path.resolve(strict=True)
        if not path.is_file():
            raise ValueError("image_path must identify a local file")
        actual_sha = sha256_file(path)
        if actual_sha.lower() != expected_media_sha256.lower():
            raise ValueError("image SHA-256 differs from expected_media_sha256")
        inputs = image_tensor(path.read_bytes()).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            feature = self.encoder.forward_features(inputs)
            if tuple(feature.shape) != (1, FEATURE_WIDTH) or not torch.isfinite(feature).all():
                raise ValueError("VideoMAE emitted invalid Clevr-4 feature")
            logits = self.scorer(
                feature.float(),
                torch.tensor([TASK_TO_INDEX[taxonomy]], dtype=torch.long, device=self.device),
            )[0]
            selected_logits = logits[option_indices].float()
            probabilities = torch.softmax(selected_logits, dim=-1).cpu().tolist()
        return {
            "probabilities": [float(value) for value in probabilities],
            "prediction_index": int(np.argmax(probabilities)),
            "confidence": float(max(probabilities)),
            "media_sha256": actual_sha,
            "feature_shape": list(feature.shape),
        }
