"""Pinned VideoMAE V2 Clevr-4 spatial-grid image Decision runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from tiny_omni_decision.dataset import CLEVR4_TAXONOMIES
from tiny_omni_decision.videomae_runtime import (
    ENCODER_CHECKPOINT_SHA256,
    ENCODER_PARAMETERS,
    ENCODER_SOURCE_SHA256,
    _load_encoder,
    image_tensor,
    sha256_file,
)
from tiny_omni_decision.videomae_spatial import (
    FEATURE_WIDTH,
    TASK_TO_INDEX,
    SpatialGridClevr4Scorer,
    videomae_spatial_features,
)

SPATIAL_READOUT_SHA256 = "10e5f02524f7ae7dd6576950cb896c61fdcb71b270bacc923b82441f85cfc5bf"
SPATIAL_READOUT_PARAMETERS = 789_962
SPATIAL_GRID_SIZE = 4
OPTION_TO_INDEX = {
    task: {str(option): index for index, option in enumerate(options)}
    for task, options in CLEVR4_TAXONOMIES.items()
}


class VideoMAESpatialGridClevr4Runtime:
    """Score supplied Clevr-4 options from the pinned 4x4 spatial readout."""

    def __init__(
        self,
        model_root: Path,
        readout_path: Path,
        dependency_root: Path,
        *,
        device: str,
    ) -> None:
        if device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("VideoMAE CUDA runtime requested but CUDA is unavailable")
        self.device = torch.device(device)
        torch.set_num_threads(4)
        self.encoder = _load_encoder(model_root.resolve(strict=True), dependency_root)
        self.encoder.to(self.device).eval().requires_grad_(False)
        if sha256_file(readout_path) != SPATIAL_READOUT_SHA256:
            raise ValueError("VideoMAE spatial-grid readout hash mismatch")
        self.scorer = SpatialGridClevr4Scorer().to(self.device).eval().requires_grad_(False)
        state = torch.load(readout_path, map_location=self.device, weights_only=True)
        self.scorer.load_state_dict(state, strict=True)
        if sum(parameter.numel() for parameter in self.scorer.parameters()) != (
            SPATIAL_READOUT_PARAMETERS
        ):
            raise ValueError("unexpected VideoMAE spatial-grid readout parameter count")

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "encoder": "OpenGVLab VideoMAE V2 ViT-Small K710",
            "encoder_source_sha256": ENCODER_SOURCE_SHA256,
            "encoder_checkpoint_sha256": ENCODER_CHECKPOINT_SHA256,
            "readout_sha256": SPATIAL_READOUT_SHA256,
            "encoder_parameters": ENCODER_PARAMETERS,
            "readout_parameters": SPATIAL_READOUT_PARAMETERS,
            "unique_parameters": ENCODER_PARAMETERS + SPATIAL_READOUT_PARAMETERS,
            "feature_width": FEATURE_WIDTH,
            "spatial_grid": [SPATIAL_GRID_SIZE, SPATIAL_GRID_SIZE],
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
            features = videomae_spatial_features(self.encoder, inputs)
            if tuple(features.shape) != (1, FEATURE_WIDTH) or not torch.isfinite(features).all():
                raise ValueError("VideoMAE emitted invalid spatial-grid feature")
            logits = self.scorer(
                features.float(),
                torch.tensor([TASK_TO_INDEX[taxonomy]], dtype=torch.long, device=self.device),
            )[0]
            probabilities = torch.softmax(logits[option_indices].float(), dim=-1).cpu().tolist()
        return {
            "probabilities": [float(value) for value in probabilities],
            "prediction_index": int(torch.tensor(probabilities).argmax().item()),
            "confidence": float(max(probabilities)),
            "media_sha256": actual_sha,
            "feature_shape": list(features.shape),
        }
