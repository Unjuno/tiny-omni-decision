from __future__ import annotations

import pytest

from scripts.audit_habit_train_text_identifiability import (
    _model_visible_text_key,
    audit_train_rows,
)


def _record(
    sample_id: str, question: str, options: list[str], target: int, *,
    split: str = "train", episode: int = 1, sid: str = "group",
) -> dict:
    return {
        "id": sample_id,
        "split": split,
        "question": question,
        "options": options,
        "target_index": target,
        "sid": sid,
        "episode_index": episode,
        "video_sha256": f"video-{episode}",
    }


def test_model_visible_key_is_option_order_invariant_and_preserves_target_text() -> None:
    first = _model_visible_text_key(
        _record("a", "Move   the block", ["left", "right"], 0)
    )
    reordered = _model_visible_text_key(
        _record("b", "MOVE the block", ["right", "left"], 1)
    )

    assert first == reordered


def test_train_audit_counts_only_train_and_finds_conflicting_repeated_inputs() -> None:
    rows = [
        _record("a", "Move the block", ["left", "right"], 0, episode=1),
        _record("b", " Move the block ", ["right", "left"], 0, episode=2),
        _record("c", "Move the block", ["left", "right"], 1, episode=3),
        _record("val", "Move the block", ["left", "right"], 1, split="validation"),
    ]

    report = audit_train_rows(rows)

    assert report["train_examples"] == 3
    assert report["unique_model_visible_text_keys"] == 1
    assert report["conflicting_text_keys"] == 1
    assert report["records_in_conflicting_text_keys"] == 3
    assert report["conflicting_target_assignments_beyond_majority"] == 1
    assert report["cross_episode_conflict_keys"] == 1
    assert report["records_in_cross_episode_conflicts"] == 3
    assert report["cross_video_conflict_keys"] == 1
    assert report["conflict_key_majority_oracle"]["accuracy"] == pytest.approx(2 / 3)
    assert report["validation_metrics_computed"] is False
    assert report["checkpoint_selection_performed"] is False


def test_train_audit_rejects_duplicate_normalized_options() -> None:
    with pytest.raises(ValueError, match="options must be unique"):
        _model_visible_text_key(
            _record("a", "Pick", ["left", " Left "], 0)
        )
