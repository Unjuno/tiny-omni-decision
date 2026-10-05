from __future__ import annotations

import pytest

from tiny_omni_decision.massive import build_massive_splits, partition_from_raw_line

REVISION = "f966f21846043aabef9b0f974fa7970027f43738"


def _row(record_id: str, partition: str, utterance: str, intent: str) -> dict[str, str]:
    return {
        "id": record_id,
        "locale": "en-US",
        "partition": partition,
        "utt": utterance,
        "intent": intent,
    }


def test_partition_prefilter_does_not_need_to_parse_reserved_test_labels() -> None:
    raw = b'{"partition":"test","intent": NOT_JSON, "utt": NOT_JSON}'
    assert partition_from_raw_line(raw) == "test"


def test_massive_preserves_full_taxonomy_and_groups_normalized_utterances() -> None:
    rows = [
        _row("1", "train", "Turn the lights off", "iot_lightoff"),
        _row("2", "dev", " turn   the lights off ", "iot_lightoff"),
        _row("3", "train", "set an alarm", "alarm_set"),
        _row("4", "dev", "play music", "music_play"),
        _row("5", "train", "set a timer", "alarm_set"),
        _row("6", "train", "create a playlist", "music_play"),
    ]
    train, validation, report = build_massive_splits(rows, revision=REVISION, seed=17)
    assert len({len(item.options) for item in train + validation}) == 1
    assert len(train[0].options) == 3
    train_groups = {item.task_group_id for item in train}
    validation_groups = {item.task_group_id for item in validation}
    assert train_groups.isdisjoint(validation_groups)
    assert report["train_validation_normalized_overlap"] == 0
    assert report["official_train_dev_utterance_groups_overlapping"] == 1
    assert report["official_train_rows_grouped_with_dev"] == 1
    assert report["official_train_rows_moved_to_validation"] == 1
    assert {item.target for item in validation} == set(train[0].options)
    assert all(item.source_record_id for item in train + validation)


def test_massive_group_assignment_is_deterministic() -> None:
    rows = [
        _row("1", "train", "alpha", "a"),
        _row("2", "train", "beta", "b"),
        _row("3", "dev", "gamma", "a"),
    ]
    first = build_massive_splits(rows, revision=REVISION, seed=23)
    second = build_massive_splits(rows, revision=REVISION, seed=23)
    assert [item.id for item in first[0]] == [item.id for item in second[0]]
    assert [item.id for item in first[1]] == [item.id for item in second[1]]
    assert first[2] == second[2]


def test_massive_refuses_unapproved_partitions_and_non_english_rows() -> None:
    with pytest.raises(ValueError, match="only train and dev"):
        build_massive_splits(
            [_row("1", "test", "not loaded", "a")], revision=REVISION
        )
    with pytest.raises(ValueError, match="en-US"):
        build_massive_splits(
            [{**_row("1", "train", "hello", "a"), "locale": "de-DE"}],
            revision=REVISION,
        )
