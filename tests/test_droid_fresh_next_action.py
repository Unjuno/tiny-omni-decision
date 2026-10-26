import pytest

from tiny_omni_decision.droid_fresh_next_action import (
    build_task_manifest,
    label_from_velocity,
    uniform_indices,
)


def _episode(index: int, steps: int = 40):
    return {
        "episode_metadata/file_path": f"/fresh/{index}",
        "episode_metadata/recording_folderpath": f"/recording/{index}",
        "step_count": steps,
        "shard_index": 500 + index,
    }


def test_next_action_uses_a_deterministic_early_window_and_following_step():
    audit = {"episodes": [_episode(0), _episode(1)]}
    from tiny_omni_decision.droid_fresh_next_action import stable_id

    ids = [stable_id(f"fresh/{index}") for index in range(2)]
    split = {
        "sealed_audit": None,
        "train": {"episode_count": 1, "episode_ids": [ids[0]]},
        "validation": {"episode_count": 1, "episode_ids": [ids[1]]},
    }
    task = build_task_manifest(
        audit, split, {ids[0]: "x+", ids[1]: "z-"}
    )
    row = task["splits"]["train"]["rows"][0]
    assert row["input_indices"] == uniform_indices(10, 8)
    assert row["target_index"] == row["input_indices"][-1] + 1
    assert row["target_index"] == 10
    assert row["target_option_id"] == "x+"


def test_fresh_next_action_rejects_group_overlap_and_invalid_velocity():
    audit = {"episodes": [_episode(0), _episode(1)]}
    from tiny_omni_decision.droid_fresh_next_action import stable_id

    ids = [stable_id(f"fresh/{index}") for index in range(2)]
    audit["episodes"][1]["episode_metadata/recording_folderpath"] = "/recording/0"
    split = {
        "sealed_audit": None,
        "train": {"episode_count": 1, "episode_ids": [ids[0]]},
        "validation": {"episode_count": 1, "episode_ids": [ids[1]]},
    }
    with pytest.raises(ValueError, match="recording-group overlap"):
        build_task_manifest(audit, split, {ids[0]: "x+", ids[1]: "z-"})
    with pytest.raises(ValueError, match="shape or target index"):
        label_from_velocity([0.0] * 6, 2, 2)


def test_fresh_next_action_excludes_short_episodes_without_moving_split_ids():
    audit = {"episodes": [_episode(0, steps=24), _episode(1, steps=40)]}
    from tiny_omni_decision.droid_fresh_next_action import stable_id

    ids = [stable_id(f"fresh/{index}") for index in range(2)]
    split = {
        "sealed_audit": None,
        "train": {"episode_count": 1, "episode_ids": [ids[0]]},
        "validation": {"episode_count": 1, "episode_ids": [ids[1]]},
    }
    task = build_task_manifest(audit, split, {ids[1]: "stop"})
    assert task["excluded_short_episode_count"] == 1
    assert task["splits"]["train"]["episode_count"] == 0
    assert task["splits"]["validation"]["episode_count"] == 1
