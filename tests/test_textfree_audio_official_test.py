from __future__ import annotations

import pytest

from scripts.evaluate_textfree_audio_official_test import (
    _speaker_group_split,
    build_test_records,
    grouped_bootstrap,
    verify_disjoint,
)
from tiny_omni_decision.dataset import SPEECH_COMMAND_LABELS

CLASSES = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]


def _row(label: str | int, name: str, speaker: str) -> dict:
    payload = b"RIFF" + b"\x00\x00\x00\x00" + b"WAVE" + name.encode()
    return {
        "label": label,
        "speaker_id": speaker,
        "file": f"{label}/{name}.wav",
        "audio": {"path": f"{label}/{name}.wav", "bytes": payload},
    }


def test_official_test_rows_keep_only_pinned_ten_keyword_classes_and_hash_ids():
    rows = [
        *[
            _row(label, f"sample-{index}", f"speaker-{index}")
            for index, label in enumerate(CLASSES)
        ],
        _row(10, "unknown", "unknown-speaker"),
    ]
    records = build_test_records(rows, SPEECH_COMMAND_LABELS)
    assert [record["target"] for record in records] == CLASSES
    assert all(len(record["id_sha256"]) == 64 for record in records)
    assert all(len(record["media_sha256"]) == 64 for record in records)


@pytest.mark.parametrize(
    "mutator, message",
    [
        (lambda rows: rows.append(rows[0]), "duplicate official test source IDs"),
        (
            lambda rows: rows.append(_row(36, "other", "other-speaker")),
            "outside the pinned class list",
        ),
    ],
)
def test_official_test_rows_reject_duplicates_or_unpinned_label_names(mutator, message):
    rows = [
        _row(label, f"sample-{index}", f"speaker-{index}") for index, label in enumerate(CLASSES)
    ]
    mutator(rows)
    with pytest.raises(ValueError, match=message):
        build_test_records(rows, SPEECH_COMMAND_LABELS)


def test_official_test_rows_reject_duplicate_audio_content():
    rows = [
        _row(label, f"sample-{index}", f"speaker-{index}") for index, label in enumerate(CLASSES)
    ]
    rows[1]["audio"]["bytes"] = rows[0]["audio"]["bytes"]
    with pytest.raises(ValueError, match="duplicate official test audio content"):
        build_test_records(rows, SPEECH_COMMAND_LABELS)


def test_official_test_requires_zero_asset_and_speaker_overlap():
    rows = [
        _row(label, f"sample-{index}", f"speaker-{index}") for index, label in enumerate(CLASSES)
    ]
    records = build_test_records(rows, SPEECH_COMMAND_LABELS)
    train = [
        {"id": "train-id", "media_sha256": "train-asset", "speaker_group_sha256": "train-speaker"}
    ]
    validation = [
        {
            "id": "validation-id",
            "media_sha256": "validation-asset",
            "speaker_group_sha256": "validation-speaker",
        }
    ]
    assert (
        verify_disjoint(records, train, validation, {"calibration-speaker"})[
            "test_train_validation_asset_overlap"
        ]
        == 0
    )
    with pytest.raises(ValueError, match="overlaps fitted/development data"):
        verify_disjoint(records, train, validation, {records[0]["speaker_group_sha256"]})


def test_frozen_calibration_split_is_deterministic_and_speaker_grouped():
    speaker_groups = ["speaker-a", "speaker-b", "speaker-a", "speaker-c", "speaker-d"]
    first = _speaker_group_split(speaker_groups)
    second = _speaker_group_split(speaker_groups)
    assert first == second
    fit, calibration = first
    assert {speaker_groups[index] for index in fit}.isdisjoint(
        {speaker_groups[index] for index in calibration}
    )


def test_speaker_group_bootstrap_is_deterministic_and_paired():
    torch = pytest.importorskip("torch")

    records = [{"speaker_group_sha256": f"speaker-{index}"} for index in range(len(CLASSES))]
    targets = torch.arange(len(CLASSES))
    logits = torch.eye(len(CLASSES)) * 5
    result = grouped_bootstrap(
        records,
        targets,
        {"legacy_baseline": logits, "uncalibrated": logits.clone()},
        repetitions=100,
        seed=17,
    )
    repeated = grouped_bootstrap(
        records,
        targets,
        {"legacy_baseline": logits, "uncalibrated": logits.clone()},
        repetitions=100,
        seed=17,
    )
    assert result == repeated
    assert result["independent_speaker_groups"] == len(CLASSES)
    assert result["low_power_below_200_speakers"]
    assert result["paired_difference_95_percent_intervals_vs_legacy_baseline"]["uncalibrated"][
        "accuracy"
    ] == [0.0, 0.0]
