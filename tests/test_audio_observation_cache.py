from __future__ import annotations

import hashlib

from tiny_omni_decision.observation_identity import audio_observation_feature_key


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
