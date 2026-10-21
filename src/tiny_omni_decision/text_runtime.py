"""Task-scoped frozen-feature runtime for the synthetic typed-text probe."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import torch
from torch import Tensor
from transformers import AutoModel, AutoTokenizer

from tiny_omni_decision.decision import DecisionOutput, FrozenFeatureCandidateScorer

MODEL_REVISION = "4ca70771034acceecb2e72475f72050fcdde4ddc"
MODEL_WEIGHTS_SHA256 = "cf1e4e2d420c664973037c3c73125d7a8fc69952495093ef8f50596f8943a433"
READOUT_SHA256 = "12d8278cb0c13c384216ba706c4edd39a30ce6888f6a1900ea7ce26304bf02d0"
TEXT_FEATURE_SIZE = 384
PAIR_FEATURE_SIZE = 4 * TEXT_FEATURE_SIZE
READOUT_PARAMETERS = 196_865
MODEL_PARAMETERS = 17_389_824
TOTAL_PARAMETERS = 17_586_689


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def candidate_features(context: Tensor, options: Tensor) -> Tensor:
    """Build the same [context, option, |difference|, product] features as training."""
    if context.shape != (TEXT_FEATURE_SIZE,):
        raise ValueError(f"context feature must have size {TEXT_FEATURE_SIZE}")
    if options.ndim != 2 or options.shape[1] != TEXT_FEATURE_SIZE or options.shape[0] < 2:
        raise ValueError("option features must have shape [n>=2, 384]")
    if not torch.isfinite(context).all() or not torch.isfinite(options).all():
        raise ValueError("text features must be finite")
    contexts = context.unsqueeze(0).expand(len(options), -1)
    features = torch.cat(
        [contexts, options, (contexts - options).abs(), contexts * options], dim=-1
    )
    if features.shape != (len(options), PAIR_FEATURE_SIZE):
        raise AssertionError("unexpected text candidate feature shape")
    return features


class TypedTextDecisionRuntime:
    """Load the frozen MiniLM and validation-selected synthetic text readout.

    `model_dir` may point to the shared MiniLM asset in the Audio component.
    This task's measured domain is Typed Decisions Synth; arbitrary text
    decision quality is not established by this runtime.
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

        weights = model_dir / "model.safetensors"
        if not weights.is_file() or not readout_path.is_file():
            raise FileNotFoundError(weights if not weights.is_file() else readout_path)
        if sha256_file(weights) != MODEL_WEIGHTS_SHA256:
            raise ValueError("MiniLM model weight hash mismatch")
        if sha256_file(readout_path) != READOUT_SHA256:
            raise ValueError("typed-text readout hash mismatch")

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.encoder = AutoModel.from_pretrained(
            model_dir, local_files_only=True, dtype=torch.float32
        ).eval().to(self.device)
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)

        self.scorer = FrozenFeatureCandidateScorer(TEXT_FEATURE_SIZE * 4, hidden_size=128)
        state = torch.load(readout_path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict):
            raise ValueError("text readout must contain a state dictionary")
        self.scorer.load_state_dict(state, strict=True)
        self.scorer.eval().to(self.device)
        for parameter in self.scorer.parameters():
            parameter.requires_grad_(False)
        self.parameter_count = sum(
            parameter.numel()
            for model in (self.encoder, self.scorer)
            for parameter in model.parameters()
        )
        if self.parameter_count != TOTAL_PARAMETERS:
            raise ValueError(f"unexpected text runtime parameter count: {self.parameter_count}")

    def _encode(self, texts: Sequence[str]) -> Tensor:
        encoded = self.tokenizer(
            list(texts),
            max_length=256,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            hidden = self.encoder(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = ((hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)).float()
        if pooled.shape != (len(texts), TEXT_FEATURE_SIZE) or not torch.isfinite(pooled).all():
            raise ValueError("MiniLM produced invalid pooled text features")
        return pooled

    def decide(
        self, state: str, question: str, options: Sequence[str]
    ) -> DecisionOutput:
        """Return probabilities over the supplied variable-size option set."""
        if not state.strip() or not question.strip():
            raise ValueError("state and question must be nonempty")
        if len(options) < 2 or any(not option.strip() for option in options):
            raise ValueError("at least two nonempty candidate options are required")
        if len(set(options)) != len(options):
            raise ValueError("candidate options must be unique")
        texts = [f"State: {state}\nQuestion: {question}"]
        texts.extend(f"Candidate answer: {option}" for option in options)
        pooled = self._encode(texts)
        features = candidate_features(pooled[0], pooled[1:])
        with torch.inference_mode():
            logits = self.scorer(features).float()
            if not torch.isfinite(logits).all():
                raise ValueError("text readout produced non-finite option scores")
            probabilities = torch.softmax(logits, dim=-1).cpu()
            predicted = int(probabilities.argmax().item())
        return DecisionOutput(
            option_logits=logits.cpu(),
            probabilities=probabilities,
            predicted_index=predicted,
            confidence=float(probabilities[predicted].item()),
        )
