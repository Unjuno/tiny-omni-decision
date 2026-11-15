"""Local CPU runtime for the frozen MiniWoB DINOv2 screen-action candidate."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from tiny_omni_decision.miniwob_dino_readout import MiniWoBOptionScorer

MODEL_SIZE = 224
PATCH_GRID = 16
FEATURE_WIDTH = 1154
EXPECTED_ENCODER_PARAMETERS = 22_056_576
EXPECTED_ENCODER_HASHES = {
    "README.md": "4c20dca454a8e5c670e8de5c7e6040f512aeca5438516f7623eedc4e3b00599c",
    "config.json": "1809f83e3bdb1609a501a610ad4a742f4fd8ae44d72ca4aa0df52d1f2ac8628d",
    "preprocessor_config.json": "14e780d86fa1861f8751f868d7f45425b5feb55c38ca26f152ca5097ab30f828",
    "model.safetensors": "ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class MiniWoBDinoRuntime:
    """Run a pinned frozen DINOv2-small encoder and readout for supplied actions."""

    def __init__(self, encoder: Any, scorer: MiniWoBOptionScorer) -> None:
        if any(parameter.requires_grad for parameter in encoder.parameters()):
            raise ValueError("vision encoder parameters must remain frozen")
        self.encoder = encoder.to(device="cpu", dtype=torch.float32).eval()
        self.scorer = scorer.to(device="cpu", dtype=torch.float32).eval()

    @classmethod
    def from_local(
        cls,
        encoder_dir: Path,
        readout_checkpoint: Path,
    ) -> MiniWoBDinoRuntime:
        """Load only the pinned local DINO files and a strict readout checkpoint."""
        from safetensors.torch import load_file
        from transformers import AutoModel

        encoder_dir = encoder_dir.resolve(strict=True)
        actual_hashes = {
            name: sha256_file(encoder_dir / name) for name in EXPECTED_ENCODER_HASHES
        }
        if actual_hashes != EXPECTED_ENCODER_HASHES:
            raise ValueError("local DINOv2 files do not match the pinned encoder hashes")
        scorer = MiniWoBOptionScorer().cpu()
        scorer.load_state_dict(load_file(readout_checkpoint, device="cpu"), strict=True)
        encoder = AutoModel.from_pretrained(
            encoder_dir,
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
        ).to(device="cpu", dtype=torch.float32)
        parameter_count = sum(parameter.numel() for parameter in encoder.parameters())
        if parameter_count != EXPECTED_ENCODER_PARAMETERS:
            raise ValueError("loaded DINOv2 parameter count differs from pinned model")
        for parameter in encoder.parameters():
            parameter.requires_grad_(False)
        return cls(encoder, scorer)

    @torch.inference_mode()
    def encode_options(
        self,
        image: Image.Image,
        coordinates: list[tuple[float, float]],
    ) -> torch.Tensor:
        """Encode one RGB screen and normalized candidate centers as scorer features."""
        return self.encode_options_batch([image], [coordinates])[0]

    @torch.inference_mode()
    def encode_options_batch(
        self,
        images: list[Image.Image],
        coordinates_per_image: list[list[tuple[float, float]]],
    ) -> torch.Tensor:
        """Encode a batch of same-option-count screens for efficient CPU inference."""
        if not images or len(images) != len(coordinates_per_image):
            raise ValueError("nonempty images and aligned coordinate batches are required")
        option_count = len(coordinates_per_image[0])
        if option_count < 2 or any(len(coords) != option_count for coords in coordinates_per_image):
            raise ValueError("each screen must provide the same count of at least two options")
        if any(
            not math.isfinite(x)
            or not math.isfinite(y)
            or not 0 <= x <= 1
            or not 0 <= y <= 1
            for coords in coordinates_per_image
            for x, y in coords
        ):
            raise ValueError("candidate coordinates must be finite and within [0, 1]")
        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
        input_rows = []
        for image in images:
            rgb = image.convert("RGB").resize(
                (MODEL_SIZE, MODEL_SIZE), resample=Image.Resampling.BICUBIC
            )
            pixels = np.asarray(rgb, dtype=np.float32) / 255.0
            input_rows.append(((pixels - mean) / std).transpose(2, 0, 1))
        input_tensor = torch.from_numpy(np.stack(input_rows).copy())
        tokens = self.encoder(pixel_values=input_tensor).last_hidden_state
        if tokens.shape != (len(images), PATCH_GRID * PATCH_GRID + 1, 384):
            raise ValueError(f"unexpected DINOv2 token shape: {tuple(tokens.shape)}")
        patches = tokens[:, 1:].reshape(len(images), PATCH_GRID, PATCH_GRID, 384)
        shared = torch.cat((tokens[:, 0], patches.mean(dim=(1, 2))), dim=-1)
        feature_rows = []
        for batch_index, coordinates in enumerate(coordinates_per_image):
            local_features = []
            for x_norm, y_norm in coordinates:
                column = min(int(x_norm * PATCH_GRID), PATCH_GRID - 1)
                row = min(int(y_norm * PATCH_GRID), PATCH_GRID - 1)
                x0, x1 = max(0, column - 1), min(PATCH_GRID, column + 2)
                y0, y1 = max(0, row - 1), min(PATCH_GRID, row + 2)
                local_features.append(patches[batch_index, y0:y1, x0:x1].mean(dim=(0, 1)))
            xy = torch.tensor(coordinates, dtype=torch.float32)
            feature_rows.append(
                torch.cat(
                    (shared[batch_index].expand(option_count, -1), torch.stack(local_features), xy),
                    dim=-1,
                )
            )
        features = torch.stack(feature_rows)
        if (
            features.shape != (len(images), option_count, FEATURE_WIDTH)
            or not torch.isfinite(features).all()
        ):
            raise ValueError("DINOv2 option features are invalid")
        return features

    @torch.inference_mode()
    def decide(
        self,
        image: Image.Image,
        option_ids: list[str],
        coordinates: list[tuple[float, float]],
        *,
        temperature: float = 1.0,
    ) -> dict[str, Any]:
        if len(option_ids) != len(coordinates):
            raise ValueError("one coordinate pair is required for each option ID")
        return self.decide_batch(
            [image], [option_ids], [coordinates], temperature=temperature
        )[0]

    @torch.inference_mode()
    def decide_batch(
        self,
        images: list[Image.Image],
        option_ids_per_image: list[list[str]],
        coordinates_per_image: list[list[tuple[float, float]]],
        *,
        temperature: float = 1.0,
    ) -> list[dict[str, Any]]:
        if len(images) != len(option_ids_per_image) or len(images) != len(coordinates_per_image):
            raise ValueError("images, option IDs, and coordinates must have equal batch size")
        features = self.encode_options_batch(images, coordinates_per_image)
        return [
            self.scorer.decide_json(
                features[index : index + 1], option_ids, temperature=temperature
            )
            for index, option_ids in enumerate(option_ids_per_image)
        ]
