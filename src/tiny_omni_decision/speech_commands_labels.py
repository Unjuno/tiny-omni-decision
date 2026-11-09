"""Pinned ClassLabel mapping for Google Speech Commands v0.02."""

from __future__ import annotations

SPEECH_COMMANDS_LABELS = (
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
)


def speech_commands_label_for_id(label_id: int) -> str:
    if type(label_id) is not int or not 0 <= label_id < len(SPEECH_COMMANDS_LABELS):
        raise ValueError("Speech Commands label ID is outside pinned v0.02 class map")
    return SPEECH_COMMANDS_LABELS[label_id]


def validate_speech_commands_target(label_id: int, source_path: str, target: str) -> None:
    expected = speech_commands_label_for_id(label_id)
    path_class = source_path.split("/", maxsplit=1)[0]
    if target != expected or path_class != expected:
        raise ValueError("target, numeric ClassLabel, and source path class disagree")
