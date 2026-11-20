import copy
import hashlib
import json

import pytest

from tiny_omni_decision.r2m_split import (
    SPLIT_ALGORITHM,
    freeze_audio_vibration_participant_split,
)


def test_participant_split_is_fixed_disjoint_and_order_independent():
    ids = list(range(1, 15))
    first = freeze_audio_vibration_participant_split(ids)
    second = freeze_audio_vibration_participant_split(list(reversed(ids)))
    assert first == second
    assert first["seed"] == 17
    assert first["algorithm"] == SPLIT_ALGORITHM
    assert {name: len(values) for name, values in first["splits"].items()} == {
        "train": 8, "calibration": 2, "validation": 2, "sealed_audit": 2,
    }
    flattened = [value for values in first["splits"].values() for value in values]
    assert sorted(flattened) == ids
    assert len(set(flattened)) == 14
    assert first["sealed_audit_accessed"] is False
    assert first["signal_or_label_contents_read"] is False


def test_seed_hash_rule_and_manifest_hash_are_reproducible():
    result = freeze_audio_vibration_participant_split(range(1, 15))
    by_id = {row["participant_id"]: row for row in result["assignments"]}
    expected = hashlib.sha256(
        b"r2m-zenodo-audio-vibration-match-v1|seed=17|participant=1"
    ).hexdigest()
    assert by_id[1]["group_hash"] == expected
    manifest_hash = result.pop("manifest_sha256")
    encoded = json.dumps(result, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True).encode("utf-8")
    assert hashlib.sha256(encoded).hexdigest() == manifest_hash


def test_no_activity_sentinel_is_not_counted_as_a_participant():
    result = freeze_audio_vibration_participant_split([999, *range(1, 15)])
    assert result["participant_count"] == 14
    assert 999 not in [value for split in result["splits"].values() for value in split]


@pytest.mark.parametrize("ids", [
    range(1, 14),
    range(1, 16),
    [*range(1, 14), 1],
    [*range(1, 14), True],
    [*range(1, 14), "participant-14"],
])
def test_invalid_inventory_fails_closed(ids):
    with pytest.raises(ValueError):
        freeze_audio_vibration_participant_split(ids)


def test_frozen_manifest_is_not_mutated_by_returned_values():
    first = freeze_audio_vibration_participant_split(range(1, 15))
    expected = copy.deepcopy(first)
    first["splits"]["train"].clear()
    second = freeze_audio_vibration_participant_split(range(1, 15))
    assert second == expected
