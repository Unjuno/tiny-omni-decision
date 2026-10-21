"""Task-scoped local runtime for the frozen CLEVR-4 VideoPrism image probe.

This runtime supports only the four CLEVR-4 taxonomies and their pinned choice
vocabularies. It is not a general image or video decision model.
"""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CLEVR4_TAXONOMIES = {
    "color": [
        "gray",
        "red",
        "blue",
        "green",
        "brown",
        "purple",
        "cyan",
        "yellow",
        "pink",
        "orange",
    ],
    "texture": [
        "rubber",
        "metal",
        "checkered",
        "emojis",
        "wave",
        "brick",
        "star",
        "circles",
        "zigzag",
        "chessboard",
    ],
    "count": [str(value) for value in range(1, 11)],
    "shape": [
        "cube",
        "sphere",
        "monkey",
        "cone",
        "torus",
        "star",
        "teapot",
        "diamond",
        "gear",
        "cylinder",
    ],
}

MODEL_REVISION = "be719a406d563b66f0ac969e7c94bab8e997c81a"
SOURCE_COMMIT = "c6eba0ece3dd49a24ffc44582966997a4a59f1d8"
CHECKPOINT_SHA256 = "61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29"
READOUT_SHA256 = "d8499747b8b94de20972c2c9ddf6c15911727a62e1aaeea393d76f6359bb40b1"
EMBEDDINGS_SIZE = 384
IMAGE_FEATURE_SIZE = 768
HIDDEN_SIZE = 128


@dataclass(frozen=True)
class ImageDecision:
    options: tuple[str, ...]
    logits: np.ndarray
    probabilities: np.ndarray


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def candidate_features(
    image_features: np.ndarray,
    option_embeddings: np.ndarray,
) -> np.ndarray:
    """Build the image-probe [VideoPrism feature | MiniLM option] input."""
    image = np.asarray(image_features, dtype=np.float32)
    options = np.asarray(option_embeddings, dtype=np.float32)
    if image.shape != (IMAGE_FEATURE_SIZE,):
        raise ValueError(f"image features must have size {IMAGE_FEATURE_SIZE}")
    if options.ndim != 2 or options.shape[1] != EMBEDDINGS_SIZE or options.shape[0] < 2:
        raise ValueError(f"option embeddings must have shape [n>=2, {EMBEDDINGS_SIZE}]")
    if not np.isfinite(image).all() or not np.isfinite(options).all():
        raise ValueError("features must be finite")
    repeated = np.broadcast_to(image, (options.shape[0], IMAGE_FEATURE_SIZE))
    return np.concatenate((repeated, options), axis=-1)


def score_candidates(
    features: np.ndarray,
    weights: dict[str, np.ndarray],
) -> np.ndarray:
    """Apply the saved Linear-GELU-Linear scorer using exact GELU semantics."""
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != IMAGE_FEATURE_SIZE + EMBEDDINGS_SIZE:
        raise ValueError("candidate feature matrix has an invalid shape")
    hidden = values @ weights["linear1_weight"].T + weights["linear1_bias"]
    # Match torch.nn.GELU(approximate='none') using the erf definition.
    import jax
    import jax.numpy as jnp

    hidden = jax.nn.gelu(jnp.asarray(hidden), approximate=False)
    logits = hidden @ weights["linear2_weight"].T + weights["linear2_bias"]
    result = np.asarray(logits, dtype=np.float32).reshape(-1)
    if not np.isfinite(result).all():
        raise ValueError("candidate scorer produced non-finite logits")
    return result


class VideoPrismClevr4Runtime:
    """Encode one local image and score supplied choices in a CLEVR-4 taxonomy."""

    def __init__(self, bundle_dir: Path, source_root: Path) -> None:
        self.bundle_dir = bundle_dir.resolve(strict=True)
        self.source_root = source_root.resolve(strict=True)
        checkpoint = self.bundle_dir / "videoprism-base-f16r288.npz"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        if sha256_file(checkpoint) != CHECKPOINT_SHA256:
            raise ValueError("VideoPrism checkpoint hash mismatch")
        weights_path = self.bundle_dir / "readout.npz"
        embeddings_path = self.bundle_dir / "option-embeddings.npz"
        import json

        self.manifest = json.loads(
            (self.bundle_dir / "bundle-manifest.json").read_text(encoding="utf-8")
        )
        if self.manifest.get("bundle_id") != "path-a-image-videoprism-clevr4-runtime-v0":
            raise ValueError("unexpected VideoPrism runtime bundle ID")
        for path, key in (
            (weights_path, "readout"),
            (embeddings_path, "option_embeddings"),
        ):
            if sha256_file(path) != self.manifest[key]["export_sha256"]:
                raise ValueError(f"{key} export hash mismatch")
        with np.load(weights_path, allow_pickle=False) as archive:
            self.weights = {key: archive[key].copy() for key in archive.files}
        with np.load(embeddings_path, allow_pickle=False) as archive:
            self.option_embeddings = {key: archive[key].copy() for key in archive.files}
        self._load_encoder(checkpoint)

    def _load_encoder(self, checkpoint: Path) -> None:
        import sys

        import jax
        import tensorflow as tf

        source_commit = subprocess.run(
            ["git", "-C", str(self.source_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if source_commit != SOURCE_COMMIT:
            raise ValueError(f"unexpected VideoPrism source revision: {source_commit}")
        sys.path.insert(0, str(self.source_root))
        from videoprism import models as vp

        self.tf = tf
        self.jax = jax
        self.model = vp.get_model("videoprism_public_v1_base")
        variables = vp.load_pretrained_weights(
            "videoprism_public_v1_base", checkpoint_path=str(checkpoint)
        )
        self.encode = jax.jit(lambda video: self.model.apply(variables, video, train=False))

    def encode_image(self, image_path: Path) -> np.ndarray:
        return self.encode_images([image_path])[0]

    def encode_images(self, image_paths: list[Path]) -> list[np.ndarray]:
        """Encode images in the fixed batch-of-eight shape used by the source run."""
        if not image_paths or len(image_paths) > 8:
            raise ValueError("encode_images requires between one and eight images")
        frames = []
        for image_path in image_paths:
            path = image_path.resolve(strict=True)
            image = self.tf.io.decode_png(self.tf.io.read_file(str(path)), channels=3)
            height, width = int(image.shape[0]), int(image.shape[1])
            scale = max(288 / height, 288 / width)
            resized = self.tf.image.resize(
                image, (round(height * scale), round(width * scale)), method="bilinear"
            )
            cropped = self.tf.image.resize_with_crop_or_pad(resized, 288, 288)
            frame = np.asarray(cropped, dtype=np.float32) / 255.0
            frames.append(np.repeat(frame[None, ...], 8, axis=0))
        while len(frames) < 8:
            frames.append(frames[-1].copy())
        video = np.stack(frames)
        result = self.encode(self.jax.numpy.asarray(video))
        self.jax.tree_util.tree_map(
            lambda value: (
                value.block_until_ready() if hasattr(value, "block_until_ready") else value
            ),
            result,
        )
        tokens = result[0] if isinstance(result, tuple) else result
        pooled = np.asarray(tokens).mean(axis=1).astype(np.float32)[: len(image_paths)]
        if pooled.shape != (len(image_paths), IMAGE_FEATURE_SIZE) or not np.isfinite(pooled).all():
            raise ValueError("VideoPrism produced invalid image features")
        return [row for row in pooled]

    def decide(self, image_path: Path, taxonomy: str, options: tuple[str, ...]) -> ImageDecision:
        return self.decide_from_features(self.encode_image(image_path), taxonomy, options)

    def decide_from_features(
        self, image_features: np.ndarray, taxonomy: str, options: tuple[str, ...]
    ) -> ImageDecision:
        if taxonomy not in CLEVR4_TAXONOMIES:
            raise ValueError(f"unknown CLEVR-4 taxonomy: {taxonomy}")
        allowed = CLEVR4_TAXONOMIES[taxonomy]
        if len(options) < 2 or len(set(options)) != len(options):
            raise ValueError("provide at least two unique options")
        if any(option not in allowed for option in options):
            raise ValueError("options must belong to the selected CLEVR-4 taxonomy")
        vectors = np.stack([self.option_embeddings[f"{taxonomy}__{option}"] for option in options])
        features = candidate_features(image_features, vectors)
        logits = score_candidates(features, self.weights)
        shifted = logits - logits.max()
        probabilities = np.exp(shifted) / np.exp(shifted).sum()
        return ImageDecision(options, logits, probabilities.astype(np.float32))
