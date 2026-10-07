"""Minimal parameter-free option readout for embedding students."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from .schema import DecisionExample

SENTENCE_SIMILARITY_PREFIX = "task: sentence similarity | query:"


def decision_query_text(state: str, question: str, *, media_token: str | None = None) -> str:
    """Build the symmetric similarity-task query without leaking any answer option."""
    content = "\n".join(part.strip() for part in (state, question) if part.strip())
    if not content:
        raise ValueError("decision query must include state or question text")
    query = f"{SENTENCE_SIMILARITY_PREFIX} {content}"
    if media_token is not None:
        if not media_token.strip():
            raise ValueError("media_token must be non-empty when provided")
        query = f"{query}\n{media_token}"
    return query


def decision_option_text(option: str) -> str:
    """Build a candidate string using the same symmetric task prefix as the query."""
    content = option.strip()
    if not content:
        raise ValueError("decision option must be non-empty")
    return f"{SENTENCE_SIMILARITY_PREFIX} {content}"


def processor_inputs_for_decision_example(
    processor: Any,
    example: DecisionExample,
    *,
    data_root: Path,
) -> Any:
    """Create pinned-processor inputs for one query and its optional local media."""
    modality_payload: dict[str, Any] = {}
    media_token = None

    if example.modality == "text":
        if example.media:
            raise ValueError(f"{example.id}: text examples cannot carry media references")
    else:
        references = [
            reference for reference in example.media if reference.kind == example.modality
        ]
        if len(references) != 1 or len(example.media) != 1:
            raise ValueError(
                f"{example.id}: expected exactly one media reference matching {example.modality}"
            )
        reference = references[0]
        if not reference.path:
            raise ValueError(f"{example.id}: media must be locally materialized ({reference.uri})")

        root = data_root.resolve()
        media_path = (root / reference.path).resolve()
        if root not in media_path.parents:
            raise ValueError(f"{example.id}: media path escapes the configured data root")
        if not media_path.is_file():
            raise FileNotFoundError(f"{example.id}: media file is missing: {media_path}")

        if example.modality == "image":
            from PIL import Image

            with Image.open(media_path) as image:
                modality_payload["images"] = [image.convert("RGB")]
            media_token = processor.image_token
        elif example.modality == "audio":
            import wave

            import numpy as np

            with wave.open(str(media_path), "rb") as audio_file:
                sample_rate = audio_file.getframerate()
                channels = audio_file.getnchannels()
                sample_width = audio_file.getsampwidth()
                if sample_rate != 16_000:
                    raise ValueError(
                        f"{example.id}: audio requires 16 kHz input, got {sample_rate} Hz"
                    )
                if sample_width != 2:
                    raise ValueError(
                        f"{example.id}: audio must be 16-bit PCM, got {sample_width * 8}-bit"
                    )
                pcm_frames = audio_file.readframes(audio_file.getnframes())
            waveform = np.frombuffer(pcm_frames, dtype=np.int16)
            if waveform.size == 0:
                raise ValueError(f"{example.id}: audio file has no PCM samples")
            waveform = waveform.astype(np.float32) / 32768.0
            if channels > 1:
                waveform = waveform.reshape(-1, channels).mean(axis=1)
            modality_payload["audio"] = [waveform]
            media_token = processor.audio_token
        else:
            modality_payload["videos"] = [str(media_path)]
            modality_payload["videos_kwargs"] = {"return_metadata": False}
            media_token = processor.video_token

    query = decision_query_text(example.state, example.question, media_token=media_token)
    return processor(text=[query], return_tensors="pt", **modality_payload)


def processor_inputs_for_options(processor: Any, options: list[str]) -> Any:
    """Encode candidate options in source order with the same task prefix."""
    if len(options) < 2:
        raise ValueError("at least two supplied options are required")
    encoded_options = [decision_option_text(option) for option in options]
    if len(set(encoded_options)) != len(encoded_options):
        raise ValueError("supplied options must be unique")
    return processor(
        text=encoded_options,
        return_tensors="pt",
    )


def mean_pool_projected_tokens(token_embeddings: Tensor, attention_mask: Tensor) -> Tensor:
    """Apply the pinned Sentence Transformers mean-pooling semantics."""
    if token_embeddings.ndim != 3:
        raise ValueError("token_embeddings must have shape [sequence, tokens, dimensions]")
    if attention_mask.ndim != 2 or attention_mask.shape != token_embeddings.shape[:2]:
        raise ValueError("attention_mask must match the first two token_embeddings dimensions")

    mask = attention_mask.to(device=token_embeddings.device, dtype=token_embeddings.dtype)
    token_counts = mask.sum(dim=1, keepdim=True)
    if torch.any(token_counts <= 0):
        raise ValueError("every sequence must contain at least one attended token")
    return (token_embeddings * mask.unsqueeze(-1)).sum(dim=1) / token_counts


def _move_tensor_inputs(model_inputs: Mapping[str, Any], *, device: torch.device) -> dict[str, Any]:
    """Move processor tensors to the model input device without changing their dtype."""
    return {
        name: value.to(device=device) if isinstance(value, Tensor) else value
        for name, value in model_inputs.items()
    }


def model_sentence_embeddings(model: Any, model_inputs: Mapping[str, Any]) -> Tensor:
    """Run an EmbeddingGemma-style encoder and apply its native pooling pipeline."""
    attention_mask = model_inputs.get("attention_mask")
    if attention_mask is None:
        raise ValueError("model_inputs must include attention_mask for native mean pooling")
    input_embeddings = getattr(model, "get_input_embeddings", None)
    if callable(input_embeddings):
        input_embedding_layer = input_embeddings()
        input_weight = getattr(input_embedding_layer, "weight", None)
    else:
        input_weight = None
    if input_weight is None:
        try:
            input_weight = next(model.parameters())
        except (AttributeError, StopIteration) as exc:
            raise ValueError("model must expose parameters to determine its input device") from exc
    if input_weight.device.type == "meta":
        raise ValueError("model input embeddings must be materialized on a real device")
    model_inputs = _move_tensor_inputs(model_inputs, device=input_weight.device)
    attention_mask = model_inputs["attention_mask"]
    output = model(**model_inputs)
    token_embeddings = getattr(output, "last_hidden_state", None)
    if token_embeddings is None:
        raise ValueError("embedding model output must include last_hidden_state")

    pooled = mean_pool_projected_tokens(token_embeddings, attention_mask).float()
    if not torch.isfinite(pooled).all():
        raise ValueError("pooled embeddings must be finite")
    norms = torch.linalg.vector_norm(pooled, dim=-1, keepdim=True)
    if torch.any(norms == 0):
        raise ValueError("pooled embeddings must be nonzero")
    return pooled / norms


def supplied_option_logits(
    query_embedding: Tensor,
    option_embeddings: Tensor,
    *,
    temperature: float,
) -> Tensor:
    """Score supplied options by cosine similarity, preserving their input order."""
    if query_embedding.ndim != 1:
        raise ValueError("query_embedding must be a one-dimensional vector")
    if option_embeddings.ndim != 2 or option_embeddings.shape[0] < 2:
        raise ValueError("option_embeddings must have shape [at least two options, dimensions]")
    if option_embeddings.shape[1] != query_embedding.shape[0]:
        raise ValueError("query and option embedding dimensions must match")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if not torch.isfinite(query_embedding).all() or not torch.isfinite(option_embeddings).all():
        raise ValueError("embeddings must be finite")
    if torch.linalg.vector_norm(query_embedding) == 0 or torch.any(
        torch.linalg.vector_norm(option_embeddings, dim=-1) == 0
    ):
        raise ValueError("query and option embeddings must be nonzero")

    query = F.normalize(query_embedding.float(), dim=-1)
    options = F.normalize(option_embeddings.float(), dim=-1)
    return options @ query / temperature
