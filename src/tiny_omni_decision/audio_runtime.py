"""Local inference runtime for the frozen Speech Commands multi-query Path A probe.

This is a task-scoped development runtime, not a general speech or audio QA model.
It consumes one local mono 16 kHz PCM WAV and returns probabilities for each
provided candidate set. Quality was measured only on the four documented
Speech Commands question templates.
"""

from __future__ import annotations

import hashlib
import math
import wave
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from transformers import AutoModel, AutoTokenizer, WhisperFeatureExtractor, WhisperModel

from tiny_omni_decision.decision import DecisionOutput, FrozenFeatureCandidateScorer

AUDIO_MODEL_ID = "openai/whisper-tiny"
AUDIO_MODEL_REVISION = "169d4a4341b33bc18d8881c4b69c2e104e1cc0af"
AUDIO_WEIGHTS_SHA256 = "7ebd0e69e78190ffe1438491fa05cc1f5c1aa3a4c4db3bc1723adbb551ea2395"
TEXT_MODEL_ID = "sentence-transformers/paraphrase-MiniLM-L3-v2"
TEXT_MODEL_REVISION = "4ca70771034acceecb2e72475f72050fcdde4ddc"
TEXT_WEIGHTS_SHA256 = "cf1e4e2d420c664973037c3c73125d7a8fc69952495093ef8f50596f8943a433"
READOUT_SHA256 = "4efa7939a686ad97b38f9eec0b8171e6e62d125ba4eb373e5d83e33ac4df42c7"
AUDIO_FEATURE_SIZE = 384
TEXT_FEATURE_SIZE = 384
PAIR_FEATURE_SIZE = 7 * AUDIO_FEATURE_SIZE
READOUT_HIDDEN_SIZE = 128


@dataclass(frozen=True)
class DecisionQuery:
    question: str
    options: tuple[str, ...]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_pcm_wav(path: Path) -> np.ndarray:
    """Read the exact input format used by the pinned Speech Commands probe."""
    with wave.open(str(path), "rb") as stream:
        if (
            stream.getnchannels() != 1
            or stream.getsampwidth() != 2
            or stream.getframerate() != 16000
            or stream.getcomptype() != "NONE"
        ):
            raise ValueError("expected uncompressed mono 16-bit 16 kHz PCM WAV")
        samples = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2").copy()
    if samples.size == 0:
        raise ValueError("audio file is empty")
    return samples.astype(np.float32) / 32768.0


def pair_candidate_features(audio: Tensor, question: Tensor, candidates: Tensor) -> Tensor:
    """Build the feature layout used by the trained 2688-input Path A readout."""
    if audio.ndim != 1 or question.shape != audio.shape or candidates.ndim != 2:
        raise ValueError("audio, question, and options must be compatible feature vectors")
    if audio.numel() != AUDIO_FEATURE_SIZE or candidates.shape[-1] != audio.numel():
        raise ValueError(f"audio and text features must have size {AUDIO_FEATURE_SIZE}")
    if candidates.shape[0] < 2:
        raise ValueError("at least two candidate options are required")
    if not torch.isfinite(audio).all() or not torch.isfinite(question).all() or not torch.isfinite(
        candidates
    ).all():
        raise ValueError("candidate feature inputs must be finite")
    audio_rows = audio.unsqueeze(0).expand(len(candidates), -1)
    question_rows = question.unsqueeze(0).expand(len(candidates), -1)
    features = torch.cat(
        [
            audio_rows,
            question_rows,
            candidates,
            (question_rows - candidates).abs(),
            (audio_rows - candidates).abs(),
            question_rows * candidates,
            audio_rows * candidates,
        ],
        dim=-1,
    )
    if features.shape != (len(candidates), PAIR_FEATURE_SIZE):
        raise AssertionError("unexpected Path A audio readout feature shape")
    return features


class SpeechCommandsMultiQueryRuntime:
    """Run the saved frozen-feature readout on local audio and supplied choices.

    `model_dir` must contain the exact local Whisper Tiny and MiniLM snapshots;
    `readout_path` is the existing validation-selected checkpoint. No downloads,
    training, or calibration fitting are performed by this class.
    """

    def __init__(
        self,
        model_dir: Path,
        readout_path: Path,
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("runtime supports CPU or CUDA")

        audio_dir = model_dir / "audio-whisper-tiny"
        text_dir = model_dir / "text-minilm-en"
        audio_weights = audio_dir / "model.safetensors"
        text_weights = text_dir / "model.safetensors"
        for path in (audio_weights, text_weights, readout_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        if _file_sha256(audio_weights) != AUDIO_WEIGHTS_SHA256:
            raise ValueError("Whisper Tiny weight hash mismatch")
        if _file_sha256(text_weights) != TEXT_WEIGHTS_SHA256:
            raise ValueError("MiniLM weight hash mismatch")
        if _file_sha256(readout_path) != READOUT_SHA256:
            raise ValueError("Path A readout hash mismatch")

        self.audio_model = WhisperModel.from_pretrained(
            audio_dir, local_files_only=True, dtype=torch.float32
        ).eval().to(self.device)
        self.audio_encoder = self.audio_model.encoder
        self.audio_processor = WhisperFeatureExtractor.from_pretrained(
            audio_dir, local_files_only=True
        )
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_dir, local_files_only=True)
        self.text_model = AutoModel.from_pretrained(
            text_dir, local_files_only=True, dtype=torch.float32
        ).eval().to(self.device)
        for model in (self.audio_model, self.text_model):
            for parameter in model.parameters():
                parameter.requires_grad_(False)

        self.scorer = FrozenFeatureCandidateScorer(
            feature_size=PAIR_FEATURE_SIZE, hidden_size=READOUT_HIDDEN_SIZE
        ).to(self.device)
        state = torch.load(readout_path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict):
            raise ValueError("readout checkpoint must contain a state dictionary")
        self.scorer.load_state_dict(state, strict=True)
        self.scorer.eval()
        for parameter in self.scorer.parameters():
            parameter.requires_grad_(False)

        self.text_cache: dict[str, Tensor] = {}
        self.parameter_count = sum(
            parameter.numel()
            for model in (self.audio_model, self.text_model, self.scorer)
            for parameter in model.parameters()
        )
        if self.parameter_count != 55_494_785:
            raise ValueError(f"unexpected total runtime parameter count: {self.parameter_count}")

    def _encode_audio_batch(self, paths: Sequence[Path]) -> list[Tensor]:
        if not paths:
            raise ValueError("at least one audio file is required")
        waveforms = [read_pcm_wav(path) for path in paths]
        encoded = self.audio_processor(
            waveforms,
            sampling_rate=16000,
            padding="max_length",
            max_length=self.audio_processor.n_samples,
            truncation=True,
            return_tensors="pt",
        )
        input_features = encoded.input_features.to(self.device)
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.device.type == "cuda"
            else nullcontext()
        )
        with torch.inference_mode(), autocast:
            hidden = self.audio_encoder(input_features).last_hidden_state
        pooled_rows = []
        for index, waveform in enumerate(waveforms):
            valid_tokens = min(hidden.shape[1], max(1, math.ceil(len(waveform) / 320)))
            pooled = hidden[index, :valid_tokens].float().mean(dim=0).cpu()
            if pooled.shape != (AUDIO_FEATURE_SIZE,) or not torch.isfinite(pooled).all():
                raise ValueError("Whisper produced invalid pooled audio features")
            pooled_rows.append(pooled)
        return pooled_rows

    def _encode_audio(self, path: Path) -> Tensor:
        return self._encode_audio_batch([path])[0]

    def _encode_texts(self, texts: Sequence[str]) -> dict[str, Tensor]:
        missing = list(dict.fromkeys(text for text in texts if text not in self.text_cache))
        for start in range(0, len(missing), 64):
            batch = missing[start : start + 64]
            encoded = self.text_tokenizer(
                batch,
                max_length=64,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                hidden = self.text_model(**encoded).last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = ((hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)).float().cpu()
            if pooled.shape != (len(batch), TEXT_FEATURE_SIZE) or not torch.isfinite(pooled).all():
                raise ValueError("MiniLM produced invalid text features")
            self.text_cache.update(
                (text, vector) for text, vector in zip(batch, pooled, strict=True)
            )
        return {text: self.text_cache[text] for text in texts}

    @staticmethod
    def _validate_queries(queries: Sequence[DecisionQuery]) -> None:
        if not queries:
            raise ValueError("at least one decision query is required")
        for query in queries:
            if not query.question.strip() or len(query.options) < 2:
                raise ValueError("each query needs a question and at least two options")
            if any(not option.strip() for option in query.options):
                raise ValueError("candidate options cannot be empty")
            if len(set(query.options)) != len(query.options):
                raise ValueError("candidate options must be unique")

    def _score_queries(
        self,
        audio: Tensor,
        queries: Sequence[DecisionQuery],
        text_vectors: dict[str, Tensor],
    ) -> list[DecisionOutput]:
        outputs = []
        with torch.inference_mode():
            for query in queries:
                question_vector = text_vectors[f"Question: {query.question}"]
                candidate_vectors = torch.stack(
                    [text_vectors[f"Candidate answer: {option}."] for option in query.options]
                )
                features = pair_candidate_features(audio, question_vector, candidate_vectors)
                logits = self.scorer(features.to(self.device)).squeeze(-1).float()
                if not torch.isfinite(logits).all():
                    raise ValueError("readout produced non-finite candidate scores")
                probabilities = torch.softmax(logits, dim=-1).cpu()
                predicted = int(probabilities.argmax().item())
                outputs.append(
                    DecisionOutput(
                        option_logits=logits.cpu(),
                        probabilities=probabilities,
                        predicted_index=predicted,
                        confidence=float(probabilities[predicted].item()),
                    )
                )
        return outputs

    def _texts_for_queries(self, query_groups: Sequence[Sequence[DecisionQuery]]) -> list[str]:
        texts = []
        for queries in query_groups:
            for query in queries:
                texts.append(f"Question: {query.question}")
                texts.extend(f"Candidate answer: {option}." for option in query.options)
        return texts

    def decide_many(
        self, audio_path: Path, queries: Sequence[DecisionQuery]
    ) -> list[DecisionOutput]:
        """Score several questions for one audio event, reusing its encoding."""
        self._validate_queries(queries)
        text_vectors = self._encode_texts(self._texts_for_queries([queries]))
        return self._score_queries(self._encode_audio(audio_path), queries, text_vectors)

    def decide_batch(
        self,
        audio_paths: Sequence[Path],
        queries_by_audio: Sequence[Sequence[DecisionQuery]],
    ) -> list[list[DecisionOutput]]:
        """Batch separate requests; BF16 numerics depend on the encoder batch size.

        Use the same audio batch size as feature extraction when reproducing a
        cached-feature experiment. The Speech Commands reference extraction
        used batches of eight. Single-event callers can use `decide_many`.
        """
        if not audio_paths or len(audio_paths) != len(queries_by_audio):
            raise ValueError("audio path and query batch sizes must be equal and nonzero")
        for queries in queries_by_audio:
            self._validate_queries(queries)
        text_vectors = self._encode_texts(self._texts_for_queries(queries_by_audio))
        audio_vectors = self._encode_audio_batch(audio_paths)
        return [
            self._score_queries(audio, queries, text_vectors)
            for audio, queries in zip(audio_vectors, queries_by_audio, strict=True)
        ]

