"""CLEVRER binary-choice scorer for frozen VideoPrism video features.

This task module consumes an already extracted, temporally encoded VideoPrism
feature vector. Video decoding and VideoPrism execution remain in their own
runtime profile; this module does not claim raw-video end-to-end inference.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import torch
from torch import Tensor
from transformers import AutoModel, AutoTokenizer

from tiny_omni_decision.decision import DecisionOutput, FrozenFeatureCandidateScorer

VIDEO_FEATURE_SIZE = 768
TEXT_FEATURE_SIZE = 384
PAIR_FEATURE_SIZE = 4 * TEXT_FEATURE_SIZE
SCORER_FEATURE_SIZE = VIDEO_FEATURE_SIZE + PAIR_FEATURE_SIZE
VIDEO_MODEL_REVISION = "be719a406d563b66f0ac969e7c94bab8e997c81a"
VIDEO_MODEL_SHA256 = "61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29"
TEXT_MODEL_REVISION = "4ca70771034acceecb2e72475f72050fcdde4ddc"
TEXT_MODEL_SHA256 = "cf1e4e2d420c664973037c3c73125d7a8fc69952495093ef8f50596f8943a433"
READOUT_SHA256 = "c15f9184794717e5942ed45a2eec18e93b12ad9a15d4b27f8b3cdf4ff480be77"
VIDEO_MODEL_PARAMETERS = 114_365_184
TEXT_MODEL_PARAMETERS = 17_389_824
READOUT_PARAMETERS = 295_169


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def binary_candidate_features(video_feature: Tensor, question: Tensor, options: Tensor) -> Tensor:
    """Build the trained [VideoPrism | question/option pair] representation."""
    if video_feature.shape != (VIDEO_FEATURE_SIZE,):
        raise ValueError(f"video feature must have size {VIDEO_FEATURE_SIZE}")
    if question.shape != (TEXT_FEATURE_SIZE,):
        raise ValueError(f"question feature must have size {TEXT_FEATURE_SIZE}")
    if options.shape != (2, TEXT_FEATURE_SIZE):
        raise ValueError("binary option features must have shape [2, 384]")
    if not all(torch.isfinite(value).all() for value in (video_feature, question, options)):
        raise ValueError("video and text features must be finite")
    repeated = question.unsqueeze(0).expand(2, -1)
    pair = torch.cat([repeated, options, (repeated - options).abs(), repeated * options], -1)
    video = video_feature.unsqueeze(0).expand(2, -1)
    result = torch.cat([video, pair], dim=-1)
    if result.shape != (2, SCORER_FEATURE_SIZE):
        raise AssertionError("unexpected video decision feature shape")
    return result


class ClevrerBinaryVideoFeatureRuntime:
    """Score a CLEVRER candidate statement from an ordered video feature.

    The model was trained on binary examples whose prompt is the source
    question followed by a candidate statement. Its output options are
    ``wrong`` and ``correct``. Accuracy on this development validation was
    near chance; this is a runtime packaging path, not a quality claim.
    """

    options = ("wrong", "correct")

    def __init__(
        self,
        text_model_dir: Path,
        readout_path: Path,
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("runtime supports CPU or CUDA")
        weight_path = text_model_dir / "model.safetensors"
        if not weight_path.is_file() or not readout_path.is_file():
            raise FileNotFoundError(weight_path if not weight_path.is_file() else readout_path)
        if sha256_file(weight_path) != TEXT_MODEL_SHA256:
            raise ValueError("MiniLM model weight hash mismatch")
        if sha256_file(readout_path) != READOUT_SHA256:
            raise ValueError("CLEVRER VideoPrism readout hash mismatch")

        self.tokenizer = AutoTokenizer.from_pretrained(text_model_dir, local_files_only=True)
        self.encoder = AutoModel.from_pretrained(
            text_model_dir, local_files_only=True, dtype=torch.float32
        ).eval().to(self.device)
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        self.scorer = FrozenFeatureCandidateScorer(SCORER_FEATURE_SIZE, hidden_size=128)
        state = torch.load(readout_path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict):
            raise ValueError("video readout must contain a state dictionary")
        self.scorer.load_state_dict(state, strict=True)
        self.scorer.eval().to(self.device)
        for parameter in self.scorer.parameters():
            parameter.requires_grad_(False)
        self.parameter_count = sum(
            parameter.numel()
            for model in (self.encoder, self.scorer)
            for parameter in model.parameters()
        )
        if self.parameter_count != TEXT_MODEL_PARAMETERS + READOUT_PARAMETERS:
            raise ValueError(f"unexpected video scorer parameter count: {self.parameter_count}")

    def _encode(self, texts: Sequence[str], batch_size: int = 64) -> Tensor:
        if batch_size < 1:
            raise ValueError("text encoding batch size must be positive")
        pooled_rows = []
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                encoded = self.tokenizer(
                    list(texts[start : start + batch_size]),
                    max_length=128,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                ).to(self.device)
                hidden = self.encoder(**encoded).last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled_rows.append(
                    ((hidden * mask).sum(1) / mask.sum(1).clamp_min(1)).float()
                )
        pooled = torch.cat(pooled_rows) if pooled_rows else torch.empty(0, TEXT_FEATURE_SIZE)
        if pooled.shape != (len(texts), TEXT_FEATURE_SIZE) or not torch.isfinite(pooled).all():
            raise ValueError("MiniLM produced invalid pooled text features")
        return pooled

    def decide(
        self, video_feature: Tensor, question: str, candidate_statement: str
    ) -> DecisionOutput:
        """Return probabilities that one candidate statement is wrong or correct."""
        return self.decide_many(
            video_feature.unsqueeze(0), [question], [candidate_statement]
        )[0]

    def decide_many(
        self,
        video_features: Tensor,
        questions: Sequence[str],
        candidate_statements: Sequence[str],
        *,
        text_batch_size: int = 64,
    ) -> list[DecisionOutput]:
        """Batch score binary CLEVRER statements over pooled video features."""
        if video_features.ndim != 2 or video_features.shape[1] != VIDEO_FEATURE_SIZE:
            raise ValueError("video features must have shape [n, 768]")
        if len(video_features) != len(questions) or len(questions) != len(candidate_statements):
            raise ValueError("video, question, and candidate counts must match")
        if not len(video_features):
            raise ValueError("at least one video decision is required")
        if not torch.isfinite(video_features).all():
            raise ValueError("video features must be finite")
        if any(not question.strip() for question in questions) or any(
            not candidate.strip() for candidate in candidate_statements
        ):
            raise ValueError("questions and candidate statements must be nonempty")
        prompts = [
            text
            for question, statement in zip(questions, candidate_statements, strict=True)
            for text in (
                f"Question: {question}\nCandidate statement: {statement}",
                "Candidate answer: wrong",
                "Candidate answer: correct",
            )
        ]
        pooled = self._encode(prompts, batch_size=text_batch_size).reshape(
            len(questions), 3, TEXT_FEATURE_SIZE
        )
        contexts = pooled[:, 0, :]
        options = pooled[:, 1:, :]
        repeated_contexts = contexts.unsqueeze(1).expand(-1, 2, -1)
        pair = torch.cat(
            [
                repeated_contexts,
                options,
                (repeated_contexts - options).abs(),
                repeated_contexts * options,
            ],
            dim=-1,
        )
        videos = video_features.to(device=self.device, dtype=torch.float32)
        video_rows = videos.unsqueeze(1).expand(-1, 2, -1)
        features = torch.cat([video_rows, pair], dim=-1).reshape(-1, SCORER_FEATURE_SIZE)
        with torch.inference_mode():
            logits = self.scorer(features).float().reshape(len(questions), 2)
            if not torch.isfinite(logits).all():
                raise ValueError("video readout produced non-finite scores")
            probabilities = torch.softmax(logits, dim=-1).cpu()
            logits = logits.cpu()
        return [
            DecisionOutput(
                option_logits=logit,
                probabilities=probability,
                predicted_index=int(probability.argmax().item()),
                confidence=float(probability.max().item()),
            )
            for logit, probability in zip(logits, probabilities, strict=True)
        ]
