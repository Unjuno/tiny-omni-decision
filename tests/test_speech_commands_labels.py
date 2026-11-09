import pytest

from tiny_omni_decision.speech_commands_labels import (
    speech_commands_label_for_id,
    validate_speech_commands_target,
)


def test_speech_commands_v002_label_ids_follow_pinned_card_order() -> None:
    assert [speech_commands_label_for_id(i) for i in range(10)] == [
        "yes",
        "no",
        "up",
        "down",
        "left",
        "right",
        "on",
        "off",
        "stop",
        "go",
    ]


@pytest.mark.parametrize("label_id", [-1, 10, True, 1.0])
def test_speech_commands_label_map_rejects_invalid_ids(label_id: object) -> None:
    with pytest.raises(ValueError, match="label ID"):
        speech_commands_label_for_id(label_id)  # type: ignore[arg-type]


def test_speech_commands_target_must_match_numeric_id_and_path_prefix() -> None:
    validate_speech_commands_target(0, "yes/d8c314c0_nohash_0.wav", "yes")
    with pytest.raises(ValueError, match="ClassLabel"):
        validate_speech_commands_target(0, "yes/d8c314c0_nohash_0.wav", "down")
    with pytest.raises(ValueError, match="ClassLabel"):
        validate_speech_commands_target(0, "down/d8c314c0_nohash_0.wav", "yes")
