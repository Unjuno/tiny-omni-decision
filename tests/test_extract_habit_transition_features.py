from __future__ import annotations

import hashlib

import pytest

from scripts.extract_habit_transition_features import (
    _selected_records,
    verify_expected_audit_digest,
)


def _episode(split: str, episode_id: int, sid: str, video_hash: str) -> dict:
    return {
        "split": split,
        "episode_index": episode_id,
        "sid": sid,
        "task_index": episode_id,
        "video_sha256": video_hash,
        "unique_option_count": 3,
        "transition_inputs": [{
            "target_subtask_id": 4,
            "target_option_index": 1,
            "context_end_ms": 1000.0,
            "onset_ms": 1100.0,
            "video_frame_indices": list(range(8)),
            "state_row_index": 10,
            "state_age_ms": 0.0,
        }],
    }


def test_transition_record_ids_labels_and_splits_are_preserved() -> None:
    records = _selected_records({"episodes": [
        _episode("train", 11, "train-sid", "train-hash"),
        _episode("validation", 22, "validation-sid", "validation-hash"),
    ]})

    assert [row["id"] for row in records] == ["train:11:0", "validation:22:0"]
    assert [row["target_option_index"] for row in records] == [1, 1]
    assert [row["option_count"] for row in records] == [3, 3]


@pytest.mark.parametrize("same", ["sid", "media"])
def test_transition_records_reject_cross_split_group_or_media_reuse(same: str) -> None:
    train = _episode("train", 11, "same-sid", "same-hash")
    validation = _episode("validation", 22, "validation-sid", "validation-hash")
    if same == "sid":
        validation["sid"] = train["sid"]
    else:
        validation["video_sha256"] = train["video_sha256"]

    with pytest.raises(ValueError, match="overlaps train and validation"):
        _selected_records({"episodes": [train, validation]})


def test_transition_records_reject_out_of_range_targets() -> None:
    episode = _episode("train", 11, "train-sid", "train-hash")
    episode["transition_inputs"][0]["target_option_index"] = 3

    with pytest.raises(ValueError, match="outside the recorded option count"):
        _selected_records({"episodes": [episode]})


def test_scale_run_requires_the_exact_explicit_audit_hash() -> None:
    payload = b'{"source_revision":"pinned"}'
    digest = hashlib.sha256(payload).hexdigest()

    verify_expected_audit_digest(payload, digest)

    with pytest.raises(ValueError, match="explicitly pinned"):
        verify_expected_audit_digest(payload, "0" * 64)

    with pytest.raises(ValueError, match="explicitly pinned"):
        verify_expected_audit_digest(payload, "not-a-sha256")
