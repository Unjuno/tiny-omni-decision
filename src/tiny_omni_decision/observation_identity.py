"""Stable identities for cached observation-level features."""

from __future__ import annotations

import hashlib
from typing import Any

from tiny_omni_decision.cache import ObservationFeatureKey


def audio_observation_feature_key(
    row: dict[str, Any],
    *,
    encoder_revision: str,
    encoder_weights_sha256: str,
    extraction_code_sha256: str,
    preprocessor_config_sha256: str,
    preprocessing_sha256: str,
) -> ObservationFeatureKey:
    """Build a question-independent key for a pinned Whisper audio observation."""
    source = str(row["source"])
    source_id, separator, source_revision = source.rpartition("@")
    if not separator or not source_id or len(source_revision) != 40:
        raise ValueError(f"source is not pinned to a commit: {source}")
    return ObservationFeatureKey(
        modality="audio",
        source_id=source_id,
        source_revision=source_revision,
        observation_sha256=str(row["media_sha256"]),
        encoder_id="openai/whisper-tiny",
        encoder_revision=encoder_revision,
        encoder_weights_sha256=encoder_weights_sha256,
        preprocessor_id="openai/whisper-tiny/WhisperFeatureExtractor",
        preprocessor_revision=encoder_revision,
        preprocessing_sha256=preprocessing_sha256,
        feature_name="whisper_last_hidden_valid_mean",
        feature_dtype="float32",
        feature_shape=(384,),
        temporal_policy=(
            "16000hz-max_length-truncate;valid_tokens=min(hidden,ceil(samples/320));mean-v1;"
            f"extractor_config_sha256={preprocessor_config_sha256};"
            f"extraction_code_sha256={extraction_code_sha256}"
        ),
    )


def image_observation_feature_key(
    *,
    source_id: str,
    source_revision: str,
    media_sha256: str,
    encoder_id: str,
    encoder_revision: str,
    encoder_weights_sha256: str,
    preprocessor_id: str,
    preprocessor_revision: str,
    preprocessing_sha256: str,
) -> ObservationFeatureKey:
    """Build a question-independent key for a frozen image representation."""
    return ObservationFeatureKey(
        modality="image",
        source_id=source_id,
        source_revision=source_revision,
        observation_sha256=media_sha256,
        encoder_id=encoder_id,
        encoder_revision=encoder_revision,
        encoder_weights_sha256=encoder_weights_sha256,
        preprocessor_id=preprocessor_id,
        preprocessor_revision=preprocessor_revision,
        preprocessing_sha256=preprocessing_sha256,
        feature_name="mean_over_spatial_tokens",
        feature_dtype="float32",
        feature_shape=(768,),
    )


def text_observation_feature_key(
    *,
    source_id: str,
    source_revision: str,
    observation_text: str,
    encoder_id: str,
    encoder_revision: str,
    encoder_weights_sha256: str,
    tokenizer_sha256: str,
    preprocessing_sha256: str,
    feature_role: str,
    hidden_size: int,
) -> ObservationFeatureKey:
    """Build a content-addressed text feature key independent of its query context."""
    if feature_role not in {"state", "question", "candidate"}:
        raise ValueError("feature_role must be state, question, or candidate")
    if hidden_size < 1:
        raise ValueError("hidden_size must be positive")
    return ObservationFeatureKey(
        modality="text",
        source_id=source_id,
        source_revision=source_revision,
        observation_sha256=hashlib.sha256(observation_text.encode("utf-8")).hexdigest(),
        encoder_id=encoder_id,
        encoder_revision=encoder_revision,
        encoder_weights_sha256=encoder_weights_sha256,
        preprocessor_id=f"{encoder_id}/AutoTokenizer",
        preprocessor_revision=encoder_revision,
        preprocessing_sha256=preprocessing_sha256,
        feature_name=f"masked_mean_{feature_role}",
        feature_dtype="float32",
        feature_shape=(hidden_size,),
        temporal_policy=f"tokenizer_files_sha256={tokenizer_sha256};attention_mask_mean_v1",
    )
