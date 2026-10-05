from __future__ import annotations

import pytest

from tiny_omni_decision.librispeech import (
    build_librispeech_decisions,
    build_librispeech_development_decisions,
)

REVISION = "96519bc4ce8e8a57f84fd03b8833553f51e76dfc"


def _record(split: str, speaker: str, number: int, transcript: str) -> dict[str, str]:
    return {
        "split": split,
        "speaker_id": speaker,
        "chapter_id": f"chapter-{speaker}",
        "utterance_id": f"{speaker}-{number:04d}",
        "transcript": transcript,
        "media_path": f"raw/librispeech/{split}/{speaker}-{number:04d}.flac",
        "media_sha256": f"{number + 1:064x}",
    }


def _records() -> list[dict[str, str]]:
    return [
        *[
            _record("train", "train-speaker", index, f"train sentence number {index}")
            for index in range(6)
        ],
        *[
            _record("validation", "dev-speaker", index, f"dev sentence number {index}")
            for index in range(5)
        ],
    ]


def test_librispeech_builds_deterministic_four_choice_speaker_disjoint_examples() -> None:
    records = _records()
    train, validation, report = build_librispeech_decisions(records, revision=REVISION, seed=17)
    assert len(train) == 6
    assert len(validation) == 5
    assert all(
        len(item.options) == 4 and item.target in item.options
        for item in train + validation
    )
    assert all(len(set(item.options)) == 4 for item in train + validation)
    assert {item.task_group_id for item in train}.isdisjoint(
        {item.task_group_id for item in validation}
    )
    assert report["train_validation_speaker_overlap"] == 0
    assert report["train_validation_transcript_overlap"] == 0
    first = build_librispeech_decisions(records, revision=REVISION, seed=17)
    assert [item.model_dump() for item in train] == [item.model_dump() for item in first[0]]


def test_librispeech_refuses_speaker_and_content_overlap() -> None:
    records = _records()
    records[-1]["speaker_id"] = "train-speaker"
    with pytest.raises(ValueError, match="speakers cross"):
        build_librispeech_decisions(records, revision=REVISION)

    records = _records()
    records[-1]["transcript"] = records[0]["transcript"]
    with pytest.raises(ValueError, match="transcript content crosses"):
        build_librispeech_decisions(records, revision=REVISION)


def test_librispeech_rejects_non_training_or_development_splits() -> None:
    records = _records()
    records[-1]["split"] = "test"
    with pytest.raises(ValueError, match="only the official train and dev"):
        build_librispeech_decisions(records, revision=REVISION)


def test_librispeech_builds_standalone_development_partition_without_train_rows() -> None:
    records = [
        _record("validation", f"dev-speaker-{index % 2}", index, f"dev sentence {index}")
        for index in range(6)
    ]
    examples, accounting = build_librispeech_development_decisions(
        records, revision=REVISION, seed=29
    )
    assert len(examples) == 6
    assert all(item.split == "validation" and len(set(item.options)) == 4 for item in examples)
    assert accounting["unique_speakers"] == 2
    assert accounting["unique_audio_assets"] == 6
    first, _ = build_librispeech_development_decisions(records, revision=REVISION, seed=29)
    assert [item.model_dump() for item in examples] == [item.model_dump() for item in first]


def test_librispeech_standalone_development_requires_validation_rows_only() -> None:
    records = [
        _record("train", f"speaker-{index}", index, f"sentence {index}")
        for index in range(4)
    ]
    with pytest.raises(ValueError, match="must use the validation split"):
        build_librispeech_development_decisions(records, revision=REVISION)
