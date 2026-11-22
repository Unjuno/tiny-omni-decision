"""CPU runtime for the fixed spoken-color-to-screen-action pilot.

This task-specific runtime is a systems prototype. It is not a general-purpose
multimodal model or a quality claim for natural audiovisual decision tasks.
"""

from __future__ import annotations

import hashlib
import json
import math
import wave
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModel, WhisperConfig, WhisperFeatureExtractor
from transformers.models.whisper.modeling_whisper import WhisperEncoder

from tiny_omni_decision.decision import FrozenFeatureCandidateScorer

IMAGE_SIZE = 224
PATCH_GRID = 16
VISUAL_WIDTH = 1154
AUDIO_WIDTH = 384
FEATURE_WIDTH = AUDIO_WIDTH + VISUAL_WIDTH
MODEL_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
MODEL_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_pcm_wav(path: Path) -> np.ndarray:
    """Read mono 16 kHz PCM16 or IEEE float32 WAV used by the pilot."""
    if path.stat().st_size > 2_000_000:
        raise ValueError("audio file exceeds the pilot's 30-second input limit")
    return waveform_from_wav_bytes(path.read_bytes())


def waveform_from_wav_bytes(content: bytes) -> np.ndarray:
    """Decode supported WAV bytes without writing media into the model bundle."""
    if len(content) > 2_000_000:
        raise ValueError("audio bytes exceed the pilot's 30-second input limit")
    try:
        with wave.open(BytesIO(content), "rb") as stream:
            if (
                stream.getnchannels() != 1
                or stream.getsampwidth() != 2
                or stream.getframerate() != 16000
                or stream.getcomptype() != "NONE"
            ):
                raise ValueError("expected mono 16 kHz PCM16 or IEEE float32 WAV")
            raw = stream.readframes(stream.getnframes())
        waveform = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    except wave.Error as err:
        import av

        with av.open(BytesIO(content)) as container:
            if len(container.streams.audio) != 1:
                raise ValueError("expected exactly one audio stream") from err
            stream = container.streams.audio[0]
            if (
                stream.codec_context.name != "pcm_f32le"
                or stream.codec_context.channels != 1
                or stream.rate != 16000
            ):
                raise ValueError("expected mono 16 kHz PCM16 or IEEE float32 WAV") from err
            frames = list(container.decode(stream))
            if not frames or any(frame.format.name != "flt" for frame in frames):
                raise ValueError("expected nonempty IEEE float32 PCM frames") from err
            waveform = np.concatenate([frame.to_ndarray().reshape(-1) for frame in frames])
    if not waveform.size or not np.isfinite(waveform).all():
        raise ValueError("audio must contain finite waveform samples")
    if np.max(np.abs(waveform)) > 1.0001:
        raise ValueError("audio waveform magnitude exceeds normalized PCM range")
    return np.asarray(waveform, dtype=np.float32)


def _checked_bundle_file(root: Path, relative: str, expected_hash: str) -> Path:
    path = (root / relative).resolve(strict=True)
    if root.resolve() not in path.parents:
        raise ValueError("bundle file path escapes its root")
    if not path.is_file() or sha256_file(path) != expected_hash:
        raise ValueError(f"bundle file hash mismatch: {relative}")
    return path


class SpokenPaletteRuntime:
    """Frozen Whisper + dynamic-INT8 DINOv2 + calibrated option scorer."""

    def __init__(self, bundle_root: Path, *, threads: int = 4) -> None:
        self.bundle_root = bundle_root.resolve(strict=True)
        self.metadata = json.loads(
            (self.bundle_root / "runtime-bundle.json").read_text(encoding="utf-8")
        )
        if self.metadata.get("schema_version") != 1 or self.metadata.get("device") != "cpu":
            raise ValueError("unsupported runtime bundle schema or device")
        if type(threads) is not int or threads < 1:
            raise ValueError("threads must be a positive integer")
        torch.set_num_threads(threads)

        files = self.metadata.get("files")
        if not isinstance(files, dict):
            raise ValueError("bundle file hash table is missing")
        paths = {
            name: _checked_bundle_file(self.bundle_root, entry["path"], entry["sha256"])
            for name, entry in files.items()
        }
        whisper_root = self.bundle_root / "whisper"
        whisper_config = WhisperConfig.from_pretrained(whisper_root, local_files_only=True)
        whisper_config._attn_implementation = "eager"
        self.audio_encoder = WhisperEncoder(whisper_config).eval().requires_grad_(False)
        self.audio_encoder.load_state_dict(
            load_file(paths["whisper_encoder"], device="cpu"), strict=True
        )
        self.audio_processor = WhisperFeatureExtractor.from_pretrained(
            whisper_root, local_files_only=True
        )

        dino_root = self.bundle_root / "dinov2"
        dino_config = AutoConfig.from_pretrained(
            dino_root, local_files_only=True, trust_remote_code=False
        )
        float_model = AutoModel.from_config(dino_config, trust_remote_code=False).eval()
        self.visual_encoder = torch.ao.quantization.quantize_dynamic(
            float_model,
            {torch.nn.Linear},
            dtype=torch.qint8,
            inplace=False,
        ).eval()
        dynamic_state = torch.load(
            paths["dinov2_dynamic_int8_state"], map_location="cpu", weights_only=True
        )
        self.visual_encoder.load_state_dict(dynamic_state, strict=True)
        del float_model, dynamic_state

        self.scorer = FrozenFeatureCandidateScorer(FEATURE_WIDTH, hidden_size=64).eval()
        self.scorer.load_state_dict(load_file(paths["scorer"], device="cpu"), strict=True)
        for model in (self.audio_encoder, self.visual_encoder, self.scorer):
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
        normalizer = np.load(paths["normalizer"], allow_pickle=False)
        self.mean = np.asarray(normalizer["mean"], dtype=np.float32)
        self.std = np.asarray(normalizer["std"], dtype=np.float32)
        if (
            self.mean.shape != (FEATURE_WIDTH,)
            or self.std.shape != (FEATURE_WIDTH,)
            or not np.isfinite(self.mean).all()
            or not np.isfinite(self.std).all()
            or np.any(self.std <= 0)
        ):
            raise ValueError("invalid frozen training normalizer")
        self.temperature = float(self.metadata["decision"]["selected_temperature"])
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("calibration temperature must be finite and positive")
        if any(
            parameter.requires_grad
            for model in (self.audio_encoder, self.visual_encoder, self.scorer)
            for parameter in model.parameters()
        ):
            raise ValueError("all runtime model parameters must remain frozen")

    def _encode_audio(self, waveform: np.ndarray) -> np.ndarray:
        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.ndim != 1 or not waveform.size or not np.isfinite(waveform).all():
            raise ValueError("audio must be a nonempty finite mono waveform")
        if waveform.size > self.audio_processor.n_samples:
            raise ValueError("audio exceeds the Whisper 30-second input window")
        encoded = self.audio_processor(
            [waveform],
            sampling_rate=16000,
            padding="max_length",
            max_length=self.audio_processor.n_samples,
            truncation=True,
            return_tensors="pt",
        )
        with torch.inference_mode():
            hidden = self.audio_encoder(encoded.input_features).last_hidden_state
            valid_tokens = min(hidden.shape[1], max(1, math.ceil(waveform.size / 320)))
            pooled = hidden[0, :valid_tokens].float().mean(dim=0)
        if pooled.shape != (AUDIO_WIDTH,) or not torch.isfinite(pooled).all():
            raise ValueError("Whisper encoder produced invalid audio features")
        return pooled.cpu().numpy().astype(np.float32, copy=False)

    @staticmethod
    def _option_positions(options: list[dict[str, Any]]) -> list[tuple[float, float]]:
        if len(options) < 2:
            raise ValueError("at least two supplied options are required")
        positions = []
        identifiers = []
        for option in options:
            if not isinstance(option, dict):
                raise ValueError("each option must be an object")
            option_id, text = option.get("id"), option.get("text")
            x, y = option.get("x_norm"), option.get("y_norm")
            if not isinstance(option_id, str) or not option_id.strip():
                raise ValueError("option id must be a nonempty string")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("option text must be a nonempty string")
            if (
                isinstance(x, bool)
                or isinstance(y, bool)
                or not isinstance(x, (int, float))
                or not isinstance(y, (int, float))
                or not math.isfinite(float(x))
                or not math.isfinite(float(y))
                or not 0 <= float(x) <= 1
                or not 0 <= float(y) <= 1
            ):
                raise ValueError("option coordinates must be finite normalized values")
            identifiers.append(option_id)
            positions.append((float(x), float(y)))
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("option ids must be unique")
        return positions

    def _encode_visual_options(
        self, image_path: Path, positions: list[tuple[float, float]]
    ) -> np.ndarray:
        with Image.open(image_path) as source:
            rgb = source.convert("RGB").resize(
                (IMAGE_SIZE, IMAGE_SIZE), resample=Image.Resampling.BICUBIC
            )
            pixels = np.asarray(rgb, dtype=np.float32) / 255.0
        pixels = ((pixels - MODEL_MEAN) / MODEL_STD).transpose(2, 0, 1).copy()
        with torch.inference_mode():
            tokens = self.visual_encoder(pixel_values=torch.from_numpy(pixels).unsqueeze(0))
            if tokens.last_hidden_state.shape != (1, PATCH_GRID * PATCH_GRID + 1, 384):
                raise ValueError("DINOv2 returned an unexpected token shape")
            patches = tokens.last_hidden_state[:, 1:].reshape(1, PATCH_GRID, PATCH_GRID, 384)[0]
            shared = torch.cat((tokens.last_hidden_state[0, 0], patches.mean(dim=(0, 1))))
            local_rows = []
            for x_norm, y_norm in positions:
                column = min(int(x_norm * PATCH_GRID), PATCH_GRID - 1)
                row = min(int(y_norm * PATCH_GRID), PATCH_GRID - 1)
                x0, x1 = max(0, column - 1), min(PATCH_GRID, column + 2)
                y0, y1 = max(0, row - 1), min(PATCH_GRID, row + 2)
                local_rows.append(patches[y0:y1, x0:x1].mean(dim=(0, 1)))
            local = torch.stack(local_rows)
            coordinates = torch.tensor(positions, dtype=torch.float32)
            shared_rows = shared.unsqueeze(0).expand(len(positions), -1)
            result = torch.cat((shared_rows, local, coordinates), dim=-1)
        if result.shape != (len(positions), VISUAL_WIDTH) or not torch.isfinite(result).all():
            raise ValueError("DINOv2 produced invalid option features")
        return result.cpu().numpy().astype(np.float32, copy=False)

    def decide_waveform(
        self,
        waveform: np.ndarray,
        image_path: Path,
        *,
        request_id: str,
        options: list[dict[str, Any]],
    ) -> str:
        """Return typed JSON probabilities for the supplied screen-action options."""
        if not isinstance(request_id, str) or not request_id.strip():
            raise ValueError("request_id must be a nonempty string")
        positions = self._option_positions(options)
        audio = self._encode_audio(waveform)
        visual = self._encode_visual_options(image_path, positions)
        repeated_audio = np.broadcast_to(audio, (len(options), AUDIO_WIDTH))
        features = np.concatenate((repeated_audio, visual), axis=-1)
        normalized = (features - self.mean) / self.std
        if normalized.shape != (len(options), FEATURE_WIDTH) or not np.isfinite(normalized).all():
            raise ValueError("normalized candidate features are invalid")
        with torch.inference_mode():
            output = self.scorer.decide(
                torch.from_numpy(normalized.astype(np.float32, copy=False)),
                temperature=self.temperature,
            )
        return output.to_decision_json(
            request_id=request_id,
            decision_type="choice",
            option_ids=[option["id"] for option in options],
            option_texts=[option["text"] for option in options],
        )

    def decide(
        self,
        audio_path: Path,
        image_path: Path,
        *,
        request_id: str,
        options: list[dict[str, Any]],
    ) -> str:
        return self.decide_waveform(
            read_pcm_wav(audio_path), image_path, request_id=request_id, options=options
        )
