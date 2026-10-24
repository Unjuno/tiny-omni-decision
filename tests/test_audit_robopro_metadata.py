from __future__ import annotations

import json

import pytest

from scripts.audit_robopro_metadata import audit_metadata


def _write_fixture(tmp_path, *, duplicate_indices=False):
    info = {
        "total_episodes": 3,
        "total_frames": 30,
        "total_tasks": 2,
        "total_videos": 9,
        "fps": 50,
        "splits": {"train": "0:3"},
    }
    episodes = [
        {"episode_index": 0, "tasks": ["move the cup"], "length": 10},
        {"episode_index": 1, "tasks": ["move the cup"], "length": 12},
        {"episode_index": 2, "tasks": ["open the drawer"], "length": 8},
    ]
    if duplicate_indices:
        episodes[2]["episode_index"] = 1
    tasks = [
        {"task_index": 0, "task": "move the cup"},
        {"task_index": 1, "task": "open the drawer"},
    ]
    (tmp_path / "info.json").write_text(json.dumps(info), encoding="utf-8")
    (tmp_path / "episodes.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in episodes), encoding="utf-8"
    )
    (tmp_path / "tasks.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in tasks), encoding="utf-8"
    )
    return tmp_path


def test_audit_reports_prompt_groups_and_refuses_single_train_split(tmp_path):
    input_dir = _write_fixture(tmp_path)

    result = audit_metadata(
        input_dir,
        revision="fixture-revision",
        declared_duplicate_index=2,
        declared_duplicate_of=0,
    )

    assert result["observed"]["episode_rows"] == 3
    assert result["observed"]["unique_exact_prompt_groups"] == 2
    assert result["observed"]["episodes_per_exact_prompt"] == {
        "min": 1,
        "median": 1.5,
        "max": 2,
        "groups_with_one_episode": 1,
    }
    assert result["split_audit"]["declared_splits"] == {"train": "0:3"}
    assert result["split_audit"]["independent_validation_split_present"] is False
    assert result["split_audit"]["quality_evaluation_admitted"] is False
    assert result["scope"]["media_downloaded"] is False


def test_audit_rejects_duplicate_episode_indices(tmp_path):
    input_dir = _write_fixture(tmp_path, duplicate_indices=True)

    with pytest.raises(ValueError, match="episode_index values must be unique"):
        audit_metadata(
            input_dir,
            revision="fixture-revision",
            declared_duplicate_index=2,
            declared_duplicate_of=0,
        )
