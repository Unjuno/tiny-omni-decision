"""EmbeddingGemma 2 with custom mean pooling and a tiny variable-option Decision Head.

The upstream encoder remains the multimodal representation backbone. Native
sentence pooling/cosine readout is intentionally bypassed: token embeddings are
mean-pooled here, then a small shared MLP scores each supplied option.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .data import Record, media_path

MODEL_ID = "google/embeddinggemma-2"
MODEL_TYPE = "embedding_gemma2"
PREFIX = "task: classification | query: "


def validate_pin(pin: dict) -> None:
    if pin.get("model_id") != MODEL_ID:
        raise ValueError(f"student model must be {MODEL_ID}")
    if not re.fullmatch(r"[0-9a-f]{40}", str(pin.get("revision", ""))):
        raise ValueError("student revision must be an immutable 40-character commit SHA")
    if pin.get("model_type") != MODEL_TYPE or pin.get("license") != "apache-2.0":
        raise ValueError("unexpected student model type/license; inspect before using")


def _move(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move(item, device) for item in value)
    return value


def _check_lengths(features: Any, max_length: int) -> int:
    count = 0
    if isinstance(features, dict):
        for key, value in features.items():
            if key == "input_ids" and isinstance(value, torch.Tensor):
                count += 1
                if value.shape[-1] > max_length:
                    raise ValueError(f"processor sequence exceeds max_length={max_length}")
            else:
                count += _check_lengths(value, max_length)
    elif isinstance(features, (list, tuple)):
        count = sum(_check_lengths(item, max_length) for item in features)
    return count


def mean_pool(token_embeddings: Tensor, attention_mask: Tensor) -> Tensor:
    """Mean-pool unmasked token embeddings without using upstream sentence pooling."""
    if token_embeddings.ndim != 3:
        raise ValueError("token_embeddings must have shape [batch, sequence, hidden]")
    if attention_mask.ndim != 2 or attention_mask.shape != token_embeddings.shape[:2]:
        raise ValueError("attention_mask must match token embedding batch/sequence dimensions")
    if not torch.isfinite(token_embeddings).all():
        raise ValueError("token embeddings must be finite")
    mask = attention_mask.to(device=token_embeddings.device, dtype=token_embeddings.dtype)
    counts = mask.sum(dim=1, keepdim=True)
    if (counts <= 0).any():
        raise ValueError("cannot pool an input with no unmasked tokens")
    return (token_embeddings * mask.unsqueeze(-1)).sum(dim=1) / counts


class VariableOptionDecisionHead(nn.Module):
    """Small shared classifier that scores any number of supplied options independently."""

    def __init__(self, embedding_dim: int, hidden_dim: int):
        super().__init__()
        if type(embedding_dim) is not int or embedding_dim <= 0:
            raise ValueError("embedding_dim must be a positive integer")
        if type(hidden_dim) is not int or hidden_dim <= 0:
            raise ValueError("head_hidden_dim must be a positive integer")
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.input = nn.Linear(embedding_dim * 3, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, query: Tensor, options: Tensor) -> Tensor:
        if query.ndim != 1 or options.ndim != 2 or options.shape[1] != query.shape[0]:
            raise ValueError("query/options embedding shapes are incompatible")
        if options.shape[0] < 2:
            raise ValueError("at least two options are required")
        expanded = query.unsqueeze(0).expand(options.shape[0], -1)
        features = torch.cat((expanded, options, expanded * options), dim=-1)
        return self.output(F.gelu(self.input(features))).squeeze(-1)


class EmbeddingDecisionStudent(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        data_root: Path,
        *,
        max_length: int = 512,
        head_hidden_dim: int = 128,
    ):
        super().__init__()
        if type(max_length) is not int or not 1 <= max_length <= 8192:
            raise ValueError("max_length must be in 1..8192")
        if not callable(getattr(encoder, "preprocess", None)):
            raise ValueError("encoder needs Sentence Transformers 6 preprocess/forward support")
        get_dimension = getattr(encoder, "get_sentence_embedding_dimension", None)
        if not callable(get_dimension):
            raise ValueError("encoder must report its embedding dimension")
        embedding_dim = get_dimension()
        if type(embedding_dim) is not int or embedding_dim <= 0:
            raise ValueError("encoder returned an invalid embedding dimension")
        self.encoder = encoder
        self.data_root = data_root
        self.max_length = max_length
        encoder_device = next(encoder.parameters()).device
        self.decision_head = VariableOptionDecisionHead(
            embedding_dim, head_hidden_dim
        ).to(device=encoder_device)

    def ternary_exclusions(self) -> list[str]:
        """Keep the tiny Decision Head higher precision during backbone ternary QAT."""
        return [f"decision_head.{name}" for name, _ in self.decision_head.named_parameters()]

    def enable_decision_head_training(self) -> None:
        for parameter in self.decision_head.parameters():
            parameter.requires_grad_(True)

    def _embed(self, inputs: list) -> torch.Tensor:
        features = self.encoder.preprocess(
            inputs, processing_kwargs={"text": {"truncation": False}}
        )
        if not _check_lengths(features, self.max_length):
            raise ValueError("processor input_ids unavailable; cannot enforce sequence bound")
        device = next(self.encoder.parameters()).device
        moved = _move(features, device)
        if hasattr(self.encoder, "__getitem__"):
            try:
                input_module = self.encoder[0]
            except (IndexError, KeyError, TypeError) as exc:
                raise ValueError(
                    "SentenceTransformer must expose its input Transformer as module 0"
                ) from exc
        else:
            # Small test/reference input modules can be supplied directly.
            input_module = self.encoder
        output = input_module(moved)
        if not isinstance(output, dict) or "token_embeddings" not in output:
            raise ValueError(
                "input Transformer must expose token_embeddings for custom pooling"
            )
        mask = output.get("attention_mask")
        if mask is None and isinstance(moved, dict):
            mask = moved.get("attention_mask")
        if not isinstance(mask, torch.Tensor):
            raise ValueError("attention_mask is required for custom mean pooling")
        embedding = mean_pool(output["token_embeddings"].float(), mask)
        if embedding.shape[0] != len(inputs):
            raise ValueError("unexpected pooled embedding shape")
        if not torch.isfinite(embedding).all() or (embedding.norm(dim=-1) == 0).any():
            raise ValueError("non-finite or zero pooled embedding")
        return F.normalize(embedding, p=2, dim=-1)

    def forward(self, record: Record) -> torch.Tensor:
        query = dict(record.inputs)
        query["text"] = PREFIX + query["text"]
        for key in {"image", "audio", "video"} & query.keys():
            values = query[key]
            if isinstance(values, list):
                query[key] = [str(media_path(self.data_root, p)) for p in values]
            else:
                query[key] = str(media_path(self.data_root, values))
        query_vector = self._embed([query])[0]
        # Recompute options on each forward so backbone gradients remain live during QAT.
        option_vectors = self._embed([PREFIX + option for option in record.options])
        return self.decision_head(query_vector, option_vectors)


def load_student(
    pin: dict,
    *,
    data_root: Path,
    device: str,
    max_length: int = 512,
    head_hidden_dim: int = 128,
    allow_download: bool = False,
) -> EmbeddingDecisionStudent:
    validate_pin(pin)
    try:
        from sentence_transformers import SentenceTransformer
        from transformers import AutoConfig
    except ImportError as exc:
        raise RuntimeError("install requirements-student.txt in a separate environment") from exc
    config = AutoConfig.from_pretrained(
        pin["model_id"],
        revision=pin["revision"],
        trust_remote_code=False,
        local_files_only=not allow_download,
    )
    if config.model_type != MODEL_TYPE:
        raise ValueError("loaded student architecture differs from pin")
    # FP32 master parameters avoid tiny QAT updates disappearing in BF16 rounding.
    # This reference backend does NOT claim to fit any particular GPU memory budget.
    encoder = SentenceTransformer(
        pin["model_id"],
        revision=pin["revision"],
        device=device,
        trust_remote_code=False,
        local_files_only=not allow_download,
        backend="torch",
        model_kwargs={"torch_dtype": torch.float32},
    )
    return EmbeddingDecisionStudent(
        encoder,
        data_root,
        max_length=max_length,
        head_hidden_dim=head_hidden_dim,
    )
