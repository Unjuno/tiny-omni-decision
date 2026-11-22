from __future__ import annotations

import pytest

from scripts.prepare_massive_ja_intents import INTENTS, prepare_rows


def _row(
    identifier: str,
    utterance: str,
    intent: int = 0,
    *,
    partition: str = "train",
) -> dict[str, object]:
    return {
        "id": identifier,
        "locale": "ja-JP",
        "partition": partition,
        "intent": intent,
        "utt": utterance,
    }


def test_prepare_rows_is_deterministic_and_removes_normalized_text_overlap() -> None:
    train = [
        _row("source-1", "  \uFF21\u3000B "),
        _row("source-2", "train only two"),
        _row("source-3", "train only three"),
    ]
    validation = [
        _row("source-4", "a B", partition="dev"),
        _row("source-5", "validation only", 1, partition="dev"),
    ]

    first = prepare_rows(train, validation)
    second = prepare_rows(train, validation)

    assert first == second
    train_rows, validation_rows, manifest = first
    assert [row["source_id"] for row in train_rows] == ["source-2", "source-3"]
    assert len(validation_rows) == 2
    assert manifest["rows"]["train_removed_for_normalized_text_overlap"] == 1
    assert manifest["unique_slurp_id_overlap"] == 0
    assert manifest["normalized_utterance_overlap_after_removal"] == 0
    assert len(INTENTS) == 60


def test_prepare_rows_rejects_duplicate_source_ids_within_a_split() -> None:
    train = [_row("same", "質問A"), _row("same", "質問B")]

    with pytest.raises(ValueError, match="duplicate upstream IDs"):
        prepare_rows(train, [])


def test_prepare_rows_rejects_source_id_overlap_between_splits() -> None:
    train = [_row("same", "学習の質問")]
    validation = [_row("same", "検証の質問", partition="dev")]

    with pytest.raises(ValueError, match="underlying SLURP IDs overlap"):
        prepare_rows(train, validation)


def test_prepare_rows_rejects_invalid_locale_partition_and_intent() -> None:
    with pytest.raises(ValueError, match="unexpected locale or publisher partition"):
        prepare_rows([_row("id", "question", partition="dev")], [])

    with pytest.raises(ValueError, match="intent index"):
        prepare_rows([_row("id", "question", intent=60)], [])
