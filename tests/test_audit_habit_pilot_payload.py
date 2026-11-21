import pytest

from scripts.audit_habit_pilot_payload import (
    _assert_group_disjoint,
    _read_selected_episode_metadata,
    _read_selected_subtasks,
)


def _row(episode_id, *, sid, task_index, instruction):
    return {
        "episode_index": episode_id,
        "sid": sid,
        "task_index": task_index,
        "high_level_instruction": instruction,
    }


def test_group_disjoint_audit_accepts_disjoint_selected_groups(tmp_path):
    rows = [
        _row(1, sid="S1", task_index=1, instruction="Pack a box"),
        _row(2, sid="S2", task_index=2, instruction="Sort waste"),
    ]

    assert _assert_group_disjoint(rows, {1: "train", 2: "validation"}, tmp_path) == 2


def test_episode_metadata_scan_parses_only_selected_split_rows(tmp_path):
    path = tmp_path / "episodes.jsonl"
    path.write_text(
        '{"episode_index":1,"low_level_instruction":"selected"}\n'
        '{"episode_index":900,"sealed_label": INVALID_JSON}\n',
        encoding="utf-8",
    )

    rows = _read_selected_episode_metadata(path, {1})

    assert set(rows) == {1}
    assert rows[1]["low_level_instruction"] == "selected"


def test_subtask_scan_parses_only_selected_target_ids(tmp_path):
    path = tmp_path / "subtasks.jsonl"
    path.write_text(
        '{"task_index":2,"task":"target action"}\n'
        '{"task_index":901,"task": INVALID_JSON}\n',
        encoding="utf-8",
    )

    assert _read_selected_subtasks(path, {2}) == {2: "target action"}


@pytest.mark.parametrize(
    ("rows", "split_by_id", "task_details"),
    [
        (
            [
                _row(1, sid="S1", task_index=7, instruction="Same task"),
                _row(2, sid="S2", task_index=7, instruction="Same task"),
            ],
            {1: "train", 2: "validation"},
            (),
        ),
        (
            [
                _row(1, sid="S1", task_index=1, instruction="Task one"),
                _row(2, sid="S2", task_index=2, instruction="Task two"),
            ],
            {1: "train", 2: "validation"},
            ("S1_S2.md",),
        ),
        (
            [
                _row(1, sid="S1", task_index=1, instruction="Same normalized task"),
                _row(2, sid="S2", task_index=2, instruction="  SAME   normalized task "),
            ],
            {1: "train", 2: "validation"},
            (),
        ),
    ],
)
def test_group_disjoint_audit_rejects_linked_train_validation_rows(
    tmp_path, rows, split_by_id, task_details
):
    for filename in task_details:
        (tmp_path / filename).write_text("task alias", encoding="utf-8")

    with pytest.raises(ValueError, match="crosses train and validation"):
        _assert_group_disjoint(rows, split_by_id, tmp_path)
