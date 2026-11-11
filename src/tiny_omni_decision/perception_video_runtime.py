"""Frozen VideoMAE + MiniLM Decision runtime for the experimental Perception route."""

from __future__ import annotations

import hashlib
import re
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from tiny_omni_decision.perception_video_readout import PerceptionVideoDecisionReadout
from tiny_omni_decision.videomae_runtime import (
    ENCODER_CHECKPOINT_SHA256,
    ENCODER_PARAMETERS,
    ENCODER_SOURCE_SHA256,
    FEATURE_WIDTH,
    _load_encoder,
)

VIDEO_FRAME_COUNT = 16
VIDEO_FRAME_SIZE = 224
VIDEO_TOKEN_SHAPE = (8, 14, 14, FEATURE_WIDTH)
TEXT_MODEL_REVISION = "4ca70771034acceecb2e72475f72050fcdde4ddc"
TEXT_WEIGHTS_SHA256 = "cf1e4e2d420c664973037c3c73125d7a8fc69952495093ef8f50596f8943a433"
TEXT_PARAMETERS = 17_389_824
TEXT_MAX_LENGTH = 256
READOUT_SHA256 = "e9516be6d58ba82c3c4f111e59d6b724e856b3ac412bbc2c57b683d35deecb8c"
READOUT_MODULE_SHA256 = "82d8462331dce8fc1e5cf9c8d042ba5f3413039ea9e8232e965cc9375f05de84"
READOUT_PARAMETERS = 444_673
READOUT_SELECTED_EPOCH = 2
READOUT_SEED = 17
READOUT_CONFIG_SHA256 = "9d63d29fec6b56a5281ad7ec9d7ec6919b87b31a7e2fe430da7b00c2e73baa0a"
OPTION_PREFIX = re.compile(r"^[A-C]:\s*(.+)$")


def uniform_frame_indices(total: int, requested: int = VIDEO_FRAME_COUNT) -> list[int]:
    if total < requested or requested < 1:
        raise ValueError(f"video must contain at least {requested} frames")
    if requested == 1:
        return [0]
    stride = (total - 1) / (requested - 1)
    return [round(index * stride) for index in range(requested)]


def decode_video_frames(path: Path, requested: int = VIDEO_FRAME_COUNT) -> np.ndarray:
    """Decode the same endpoint-inclusive uniform RGB frames as the frozen extractor."""

    import av

    with av.open(str(path)) as container:
        streams = [stream for stream in container.streams if stream.type == "video"]
        if not streams:
            raise ValueError(f"video stream missing: {path}")
        stream = streams[0]
        total = int(stream.frames or 0)
        if total <= 0:
            if stream.duration is None or stream.time_base is None or not stream.average_rate:
                raise ValueError("video frame count cannot be derived from stream metadata")
            total = round(float(stream.duration * stream.time_base * stream.average_rate))
        wanted_list = uniform_frame_indices(total, requested)
        wanted = set(wanted_list)
        selected: dict[int, np.ndarray] = {}
        for index, frame in enumerate(container.decode(stream)):
            if index in wanted:
                selected[index] = frame.reformat(
                    width=VIDEO_FRAME_SIZE,
                    height=VIDEO_FRAME_SIZE,
                    format="rgb24",
                ).to_ndarray()
            if index >= max(wanted):
                break
    if set(selected) != wanted:
        raise ValueError(f"decoder returned {len(selected)}/{requested} frames")
    frames = np.stack([selected[index] for index in wanted_list])
    if frames.shape != (requested, VIDEO_FRAME_SIZE, VIDEO_FRAME_SIZE, 3):
        raise ValueError("decoded frames do not match the fixed VideoMAE input shape")
    if frames.dtype != np.uint8:
        raise ValueError("decoded frames must be RGB uint8")
    return frames


def _candidate_text(value: str) -> str:
    stripped = value.strip()
    match = OPTION_PREFIX.fullmatch(stripped)
    if re.fullmatch(r"[A-C]:\s*", stripped):
        raise ValueError("candidate option text must be nonempty")
    candidate = match.group(1).strip() if match else stripped
    if not candidate:
        raise ValueError("candidate option text must be nonempty")
    return candidate


def pool_candidate_queries(hidden: Tensor, attention_mask: Tensor) -> Tensor:
    """Pool three MiniLM prompts and add the reader's one-item batch axis."""

    if hidden.ndim != 3 or attention_mask.shape != hidden.shape[:2]:
        raise ValueError("expected MiniLM hidden [options,tokens,width] and matching mask")
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
    if pooled.shape != (3, FEATURE_WIDTH) or not torch.isfinite(pooled).all():
        raise ValueError("MiniLM candidate embeddings differ from the fixed 3x384 contract")
    return pooled.unsqueeze(0)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_readout_checkpoint(readout: PerceptionVideoDecisionReadout, path: Path) -> None:
    """Load the pinned training envelope while validating its selected-run identity."""

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != {
        "state_dict",
        "selected_epoch",
        "seed",
        "config_sha256",
    }:
        raise ValueError("Perception readout checkpoint envelope differs from its pinned schema")
    if (
        payload["selected_epoch"] != READOUT_SELECTED_EPOCH
        or payload["seed"] != READOUT_SEED
        or payload["config_sha256"] != READOUT_CONFIG_SHA256
        or not isinstance(payload["state_dict"], dict)
    ):
        raise ValueError("Perception readout checkpoint identity differs from its pinned run")
    readout.load_state_dict(payload["state_dict"], strict=True)


class PerceptionVideoTextRuntime:
    """Score three caller-provided candidates against one frozen video.

    The VideoMAE encoder may be shared with the existing image worker. Its
    parameters remain frozen; the added readout is the already-trained R2
    artifact and no fitting/calibration happens at runtime.
    """

    def __init__(
        self,
        *,
        model_root: Path,
        dependency_root: Path,
        text_model_dir: Path,
        readout_path: Path,
        device: str,
        video_encoder: nn.Module | None = None,
        video_cache_entries: int = 4,
    ) -> None:
        if device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("VideoMAE Perception runtime requested but CUDA is unavailable")
        if video_cache_entries < 1:
            raise ValueError("video_cache_entries must be positive")
        model_root = model_root.resolve(strict=True)
        text_model_dir = text_model_dir.resolve(strict=True)
        readout_path = readout_path.resolve(strict=True)
        self.device = torch.device(device)
        torch.set_num_threads(4)

        readout_source = Path(__file__).with_name("perception_video_readout.py")
        if _file_sha256(readout_source) != READOUT_MODULE_SHA256:
            raise ValueError("Perception readout source SHA-256 differs from the trained module")

        text_weights = text_model_dir / "model.safetensors"
        if _file_sha256(text_weights) != TEXT_WEIGHTS_SHA256:
            raise ValueError("MiniLM checkpoint SHA-256 differs from the pinned revision")
        if _file_sha256(readout_path) != READOUT_SHA256:
            raise ValueError("Perception video readout SHA-256 mismatch")

        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(text_model_dir, local_files_only=True)
        self.text_encoder = AutoModel.from_pretrained(
            text_model_dir,
            local_files_only=True,
            dtype=torch.float32,
        ).eval().requires_grad_(False).to(self.device)
        text_parameters = sum(parameter.numel() for parameter in self.text_encoder.parameters())
        if (
            text_parameters != TEXT_PARAMETERS
            or self.text_encoder.config.hidden_size != FEATURE_WIDTH
        ):
            raise ValueError(
                "MiniLM parameter count or embedding width differs from the pinned contract"
            )

        self.video_encoder = (
            video_encoder
            if video_encoder is not None
            else _load_encoder(model_root, dependency_root)
        )
        self.video_encoder.to(self.device).eval().requires_grad_(False)
        video_parameters = sum(parameter.numel() for parameter in self.video_encoder.parameters())
        if video_parameters != ENCODER_PARAMETERS:
            raise ValueError("unexpected VideoMAE parameter count")

        self.readout = PerceptionVideoDecisionReadout().to(self.device).eval().requires_grad_(False)
        load_readout_checkpoint(self.readout, readout_path)
        self.readout.to(self.device)
        if self.readout.parameter_count() != READOUT_PARAMETERS:
            raise ValueError("unexpected Perception video readout parameter count")

        self._capture: dict[str, Tensor] = {}
        self._hook = self.video_encoder.blocks[-1].register_forward_hook(self._capture_tokens)
        self._video_cache: OrderedDict[str, Tensor] = OrderedDict()
        self._video_cache_entries = video_cache_entries
        self._cache_hits = 0
        self._cache_misses = 0

    def _capture_tokens(self, _module: nn.Module, _inputs: tuple[Any, ...], output: Tensor) -> None:
        self._capture["tokens"] = output.detach()

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "profile": "path_a_videomaev2_perception_video_text_experimental",
            "video_encoder_revision": "706cc172d65ebd4dedbee3f9c0183a93df9fa125",
            "video_encoder_source_sha256": ENCODER_SOURCE_SHA256,
            "video_encoder_checkpoint_sha256": ENCODER_CHECKPOINT_SHA256,
            "video_encoder_parameters": ENCODER_PARAMETERS,
            "text_encoder_revision": TEXT_MODEL_REVISION,
            "text_encoder_weights_sha256": TEXT_WEIGHTS_SHA256,
            "text_encoder_parameters": TEXT_PARAMETERS,
            "readout_sha256": READOUT_SHA256,
            "readout_module_sha256": READOUT_MODULE_SHA256,
            "readout_parameters": READOUT_PARAMETERS,
            "readout_selected_epoch": READOUT_SELECTED_EPOCH,
            "readout_seed": READOUT_SEED,
            "readout_config_sha256": READOUT_CONFIG_SHA256,
            "resident_unique_parameters": ENCODER_PARAMETERS + TEXT_PARAMETERS + READOUT_PARAMETERS,
            "device": torch.cuda.get_device_name(self.device)
            if self.device.type == "cuda"
            else "cpu",
            "dtype": str(next(self.video_encoder.parameters()).dtype),
            "video_preprocessing": (
                "16 endpoint-inclusive uniform PyAV RGB24 frames, 224x224, /127.5 - 1"
            ),
            "text_prompt": (
                "Question: {question}\nCandidate answer: {option_text_without_A_B_C_prefix}"
            ),
            "text_pooling": "masked mean of last_hidden_state",
            "text_max_length": TEXT_MAX_LENGTH,
            "quality_metrics_computed": False,
        }

    @property
    def cache_metrics(self) -> dict[str, int]:
        return {
            "video_cache_hits": self._cache_hits,
            "video_cache_misses": self._cache_misses,
            "video_cache_entries": len(self._video_cache),
            "video_cache_limit": self._video_cache_entries,
        }

    def _video_tokens(self, path: Path, media_sha256: str) -> Tensor:
        cached = self._video_cache.get(media_sha256)
        if cached is not None:
            self._cache_hits += 1
            self._video_cache.move_to_end(media_sha256)
            return cached

        frames = decode_video_frames(path)
        values = frames.astype(np.float32) / 127.5 - 1.0
        inputs = torch.from_numpy(values.copy()).permute(3, 0, 1, 2).unsqueeze(0).to(self.device)
        self._capture.pop("tokens", None)
        with torch.inference_mode():
            pooled = self.video_encoder.forward_features(inputs)
        tokens = self._capture.pop("tokens", None)
        expected_shape = (1, int(np.prod(VIDEO_TOKEN_SHAPE[:3])), FEATURE_WIDTH)
        if tokens is None or tuple(tokens.shape) != expected_shape:
            raise ValueError("VideoMAE final-block tokens differ from the frozen training contract")
        if tuple(pooled.shape) != (1, FEATURE_WIDTH):
            raise ValueError("VideoMAE pooled output differs from the frozen training contract")
        torch.testing.assert_close(
            pooled,
            self.video_encoder.fc_norm(tokens.mean(1)),
            rtol=0,
            atol=0,
        )
        grid = tokens.reshape((1, *VIDEO_TOKEN_SHAPE)).float().detach()
        if not torch.isfinite(grid).all():
            raise FloatingPointError("VideoMAE produced non-finite token features")
        self._video_cache[media_sha256] = grid
        self._video_cache.move_to_end(media_sha256)
        while len(self._video_cache) > self._video_cache_entries:
            self._video_cache.popitem(last=False)
        self._cache_misses += 1
        return grid

    def decide(
        self,
        *,
        video_path: Path,
        question: str,
        options: list[str],
        expected_media_sha256: str,
    ) -> dict[str, Any]:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be nonempty text")
        if len(options) != 3:
            raise ValueError("this experimental readout is trained for exactly three candidates")
        candidates = [_candidate_text(option) for option in options]
        if len(set(candidates)) != len(candidates):
            raise ValueError("candidate option texts must be unique after prefix removal")
        if not re.fullmatch(r"[0-9a-fA-F]{64}", expected_media_sha256):
            raise ValueError("expected_media_sha256 must be a 64-character hex digest")
        path = video_path.resolve(strict=True)
        if not path.is_file():
            raise ValueError("video_path must identify a local file")
        actual_sha = _file_sha256(path)
        if actual_sha.lower() != expected_media_sha256.lower():
            raise ValueError("video SHA-256 differs from expected_media_sha256")

        started = time.perf_counter()
        video_grid = self._video_tokens(path, actual_sha)
        video_tokens = video_grid.reshape(1, -1, FEATURE_WIDTH)
        prompts = [
            f"Question: {question.strip()}\nCandidate answer: {candidate}"
            for candidate in candidates
        ]
        with torch.inference_mode():
            encoded = self.tokenizer(
                prompts,
                max_length=TEXT_MAX_LENGTH,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            hidden = self.text_encoder(**encoded).last_hidden_state
            queries = pool_candidate_queries(hidden, encoded["attention_mask"])
            logits = self.readout(video_tokens, queries.float())[0].float()
            if logits.shape != (3,) or not torch.isfinite(logits).all():
                raise FloatingPointError("Perception video readout produced invalid logits")
            probabilities = torch.softmax(logits, dim=-1)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        probability_values = [float(value) for value in probabilities.cpu().tolist()]
        prediction_index = max(range(3), key=probability_values.__getitem__)
        return {
            "logits": [float(value) for value in logits.cpu().tolist()],
            "probabilities": probability_values,
            "prediction_index": prediction_index,
            "confidence": probability_values[prediction_index],
            "media_sha256": actual_sha,
            "video_feature_shape": list(video_grid.shape[1:]),
            "candidate_count": len(candidates),
            "inference_seconds": time.perf_counter() - started,
            **self.cache_metrics,
        }

    def close(self) -> None:
        self._hook.remove()

