"""Local JSON inference for the frozen-feature MiniWoB DINO Path A candidate.

This runtime is specific to the synthetic ``click-color-v1:screen-only`` task.
The readout uses screenshot features and candidate coordinates; option text is
preserved in the response but is not a model input.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from tiny_omni_decision.decision_json import decision_json_record
from tiny_omni_decision.miniwob_dino_readout import MiniWoBOptionScorer

_MODEL_SIZE = 224
_PATCH_SIZE = 14
_PATCH_GRID = _MODEL_SIZE // _PATCH_SIZE
_FEATURE_WIDTH = 1154
_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_options(
    options: Sequence[Mapping[str, Any]],
) -> tuple[list[str], list[str], list[tuple[float, float]]]:
    if isinstance(options, (str, bytes)) or len(options) < 2:
        raise ValueError("at least two supplied options are required")
    ids: list[str] = []
    texts: list[str] = []
    positions: list[tuple[float, float]] = []
    for option in options:
        if not isinstance(option, Mapping) or set(option) != {"id", "text", "x_norm", "y_norm"}:
            raise ValueError("each option must contain exactly id, text, x_norm, and y_norm")
        option_id, text = option["id"], option["text"]
        if not isinstance(option_id, str) or not option_id.strip():
            raise ValueError("option IDs must be nonempty strings")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("option texts must be nonempty strings")
        coordinates: list[float] = []
        for key in ("x_norm", "y_norm"):
            value = option[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("option coordinates must be finite numbers in [0, 1]")
            value = float(value)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("option coordinates must be finite numbers in [0, 1]")
            coordinates.append(value)
        ids.append(option_id)
        texts.append(text)
        positions.append((coordinates[0], coordinates[1]))
    if len(set(ids)) != len(ids):
        raise ValueError("option IDs must be unique")
    return ids, texts, positions


def encode_screen_option_features(
    image: Image.Image,
    positions: Sequence[tuple[float, float]],
    encoder: torch.nn.Module,
) -> torch.Tensor:
    """Extract the exact CLS/global/local/xy feature layout used by Path A."""
    if not isinstance(image, Image.Image) or image.size != (160, 210):
        raise ValueError("MiniWoB screen input must be a 160x210 PIL image")
    if len(positions) < 2:
        raise ValueError("at least two candidate positions are required")
    pixels = np.asarray(
        image.convert("RGB").resize(
            (_MODEL_SIZE, _MODEL_SIZE), resample=Image.Resampling.BICUBIC
        ),
        dtype=np.float32,
    ) / 255.0
    tensor = torch.from_numpy(((pixels - _MEAN) / _STD).transpose(2, 0, 1).copy()).unsqueeze(0)
    parameter = next(encoder.parameters(), None)
    device = parameter.device if parameter is not None else tensor.device
    dtype = parameter.dtype if parameter is not None else tensor.dtype
    tensor = tensor.to(device=device, dtype=dtype)
    encoder.eval()
    with torch.inference_mode():
        output = encoder(pixel_values=tensor)
        tokens = output.last_hidden_state
        if tokens.shape != (1, _PATCH_GRID * _PATCH_GRID + 1, 384):
            raise ValueError(f"unexpected DINOv2 token shape: {tuple(tokens.shape)}")
        patches = tokens[:, 1:].reshape(1, _PATCH_GRID, _PATCH_GRID, 384)
        shared = torch.cat((tokens[:, 0], patches.mean(dim=(1, 2))), dim=1)
        local_features: list[torch.Tensor] = []
        for x_norm, y_norm in positions:
            if (
                isinstance(x_norm, bool)
                or isinstance(y_norm, bool)
                or not math.isfinite(float(x_norm))
                or not math.isfinite(float(y_norm))
                or not 0 <= x_norm <= 1
                or not 0 <= y_norm <= 1
            ):
                raise ValueError("candidate coordinates must be finite and normalized")
            column = min(int(float(x_norm) * _PATCH_GRID), _PATCH_GRID - 1)
            row = min(int(float(y_norm) * _PATCH_GRID), _PATCH_GRID - 1)
            local = patches[0, max(0, row - 1):min(_PATCH_GRID, row + 2),
                             max(0, column - 1):min(_PATCH_GRID, column + 2)].mean(dim=(0, 1))
            xy = tokens.new_tensor([x_norm, y_norm])
            local_features.append(torch.cat((shared[0], local, xy)))
        features = torch.stack(local_features).unsqueeze(0)
    if features.shape != (1, len(positions), _FEATURE_WIDTH) or not torch.isfinite(features).all():
        raise ValueError("invalid or non-finite MiniWoB option features")
    return features


class MiniWoBDinoDecisionRuntime:
    """Turn one MiniWoB screenshot and supplied action candidates into JSON."""

    def __init__(self, encoder: torch.nn.Module, scorer: MiniWoBOptionScorer) -> None:
        self.encoder = encoder.eval()
        self.scorer = scorer.eval()
        if any(parameter.requires_grad for parameter in self.encoder.parameters()):
            raise ValueError("the DINOv2 encoder must remain frozen")

    @classmethod
    def from_local_artifacts(
        cls,
        *,
        encoder_dir: Path,
        readout_path: Path,
        encoder_sha256: str,
        readout_sha256: str,
        device: str = "cpu",
    ) -> MiniWoBDinoDecisionRuntime:
        """Load exact local artifacts after hash checks; never access the network."""
        encoder_dir = encoder_dir.resolve(strict=True)
        readout_path = readout_path.resolve(strict=True)
        weight_path = encoder_dir / "model.safetensors"
        if sha256_file(weight_path) != encoder_sha256:
            raise ValueError("DINOv2 weight SHA-256 mismatch")
        if sha256_file(readout_path) != readout_sha256:
            raise ValueError("readout SHA-256 mismatch")
        from safetensors.torch import load_file
        from transformers import AutoModel

        encoder = AutoModel.from_pretrained(
            encoder_dir,
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
        ).to(device=device, dtype=torch.float32)
        for parameter in encoder.parameters():
            parameter.requires_grad_(False)
        if sum(parameter.numel() for parameter in encoder.parameters()) != 22_056_576:
            raise ValueError("loaded DINOv2 parameter count differs from the pinned small model")
        scorer = MiniWoBOptionScorer().to(device=device, dtype=torch.float32)
        state = load_file(str(readout_path), device=device)
        scorer.load_state_dict(state, strict=True)
        return cls(encoder, scorer)

    def predict(
        self,
        *,
        request_id: str,
        image: Image.Image,
        options: Sequence[Mapping[str, Any]],
    ) -> dict[str, object]:
        option_ids, option_texts, positions = _validate_options(options)
        features = encode_screen_option_features(image, positions, self.encoder)
        parameter = next(self.scorer.parameters())
        with torch.inference_mode():
            logits = self.scorer(features.to(device=parameter.device, dtype=parameter.dtype))
            probabilities = torch.softmax(logits.float(), dim=-1)[0].cpu().tolist()
        return decision_json_record(
            request_id=request_id,
            decision_type="choice",
            option_ids=option_ids,
            option_texts=option_texts,
            probabilities=probabilities,
        )
