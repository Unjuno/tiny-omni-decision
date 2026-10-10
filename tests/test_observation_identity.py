from __future__ import annotations

import hashlib

from tiny_omni_decision.observation_identity import (
    audio_observation_feature_key,
    image_observation_feature_key,
    text_observation_feature_key,
)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _key(row: dict[str, str]):
    return audio_observation_feature_key(
        row,
        encoder_revision="169d4a4341b33bc18d8881c4b69c2e104e1cc0af",
        encoder_weights_sha256=_sha256(b"whisper weights"),
        extraction_code_sha256=_sha256(b"extraction code"),
        preprocessor_config_sha256=_sha256(b"preprocessor config"),
        preprocessing_sha256=_sha256(b"preprocessing spec"),
    )


def test_audio_observation_cache_key_is_question_independent_and_asset_scoped():
    base = {
        "source": "google/speech_commands@a751309c0fd613e8a5d30d77900f30e8b42bc2da",
        "media_sha256": _sha256(b"audio asset one"),
        "question": "Which word was spoken?",
        "options": ["yes", "no"],
    }

    same_observation_other_query = {
        **base,
        "question": "Is a response spoken?",
        "options": ["response", "not response"],
    }
    other_observation = {**base, "media_sha256": _sha256(b"audio asset two")}

    assert _key(base).cache_id == _key(same_observation_other_query).cache_id
    assert _key(base).cache_id != _key(other_observation).cache_id


def _image_key(media: bytes, preprocessing: bytes = b"preprocessing spec"):
    return image_observation_feature_key(
        source_id="sgvaze/clevr4",
        source_revision="cddc78fb2a8359dc958987b2c750bfdd4bfd2c73",
        media_sha256=_sha256(media),
        encoder_id="facebookresearch/vjepa2/vit_base_16_384",
        encoder_revision="204698b45b3712590f06245fbfba32d3be539812",
        encoder_weights_sha256=_sha256(b"vjepa weights"),
        preprocessor_id="official_vjepa2_preprocessor_384",
        preprocessor_revision="204698b45b3712590f06245fbfba32d3be539812",
        preprocessing_sha256=_sha256(preprocessing),
    )


def test_image_observation_cache_key_invalidates_on_asset_or_preprocessor_change():
    original = _image_key(b"image asset")

    assert original.cache_id == _image_key(b"image asset").cache_id
    assert original.cache_id != _image_key(b"different image asset").cache_id
    assert original.cache_id != _image_key(b"image asset", b"different preprocessing").cache_id


def _text_key(text: str, *, role: str = "state", preprocessing: bytes = b"tokenizer policy"):
    return text_observation_feature_key(
        source_id="n4ze3m/typed-decisions-synth",
        source_revision="5ece89a225b23c4cd5c4bab5735a0819d61dd7d5",
        observation_text=text,
        encoder_id="sentence-transformers/paraphrase-MiniLM-L3-v2",
        encoder_revision="4ca70771034acceecb2e72475f72050fcdde4ddc",
        encoder_weights_sha256=_sha256(b"minilm weights"),
        tokenizer_sha256=_sha256(b"tokenizer files"),
        preprocessing_sha256=_sha256(preprocessing),
        feature_role=role,
        hidden_size=384,
    )


def test_text_state_feature_key_is_reusable_across_questions_and_asset_scoped():
    first_question_state = _text_key("State: room temperature is 20 C")
    second_question_state = _text_key("State: room temperature is 20 C")

    assert first_question_state.cache_id == second_question_state.cache_id
    assert first_question_state.cache_id != _text_key("State: room temperature is 21 C").cache_id


def test_text_feature_key_invalidates_on_role_or_tokenization_change():
    original = _text_key("Question: which value?", role="question")

    assert original.cache_id != _text_key("Question: which value?", role="state").cache_id
    assert (
        original.cache_id
        != _text_key(
            "Question: which value?", role="question", preprocessing=b"changed tokenizer policy"
        ).cache_id
    )
