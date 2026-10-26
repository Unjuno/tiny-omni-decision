import pytest

from tiny_omni_decision.droid_early_outcome import (
    build_early_outcome_rows,
    canonical_path,
    source_outcome,
    stable_id,
    uniform_indices,
)


def _episode(path, group, *, steps=40, shard=110):
    return {
        "episode_metadata/file_path": path,
        "episode_metadata/recording_folderpath": group,
        "step_count": steps,
        "shard_index": shard,
    }


def _audit_and_split():
    episodes = [
        _episode("/dataset/success/day/a", "/recording/a"),
        _episode("/dataset/failure/day/b", "/recording/b"),
    ]
    ids = [stable_id(canonical_path(row["episode_metadata/file_path"])) for row in episodes]
    audit = {"episodes": episodes}
    split = {
        "sealed_audit": None,
        "train": {"episode_count": 1, "episode_ids": [ids[0]]},
        "validation": {"episode_count": 1, "episode_ids": [ids[1]]},
    }
    return audit, split


def test_uniform_observation_indices_are_early_and_unique():
    assert uniform_indices(10, 8) == [0, 1, 3, 4, 5, 6, 8, 9]
    assert uniform_indices(8, 8) == list(range(8))
    with pytest.raises(ValueError, match="cover"):
        uniform_indices(7, 8)


def test_source_outcome_requires_one_exact_path_segment():
    assert source_outcome("/dataset/SUCCESS/day/episode") == "success"
    assert source_outcome("/dataset/failure/day/episode") == "failure"
    with pytest.raises(ValueError, match="exactly one"):
        source_outcome("/dataset/unsuccessful/day/episode")
    with pytest.raises(ValueError, match="exactly one"):
        source_outcome("/dataset/success/failure/episode")


def test_task_manifest_is_group_safe_and_uses_only_first_quarter():
    audit, split = _audit_and_split()
    result = build_early_outcome_rows(audit, split)
    train = result["splits"]["train"]["rows"][0]
    validation = result["splits"]["validation"]["rows"][0]
    assert train["target_option_id"] == "success"
    assert validation["target_option_id"] == "failure"
    assert train["options"] == ["success", "failure"]
    assert train["observation_indices"] == [0, 1, 3, 4, 5, 6, 8, 9]
    assert max(train["observation_indices"]) < train["observed_step_count"] == 10
    assert train["group_id"] != validation["group_id"]
    assert result["splits"]["train"]["target_counts"] == {"success": 1, "failure": 0}
    assert result["splits"]["validation"]["target_counts"] == {"success": 0, "failure": 1}
    assert result["input_contract"]["future_observations_in_input"] is False
    assert result["input_contract"]["timebase"].startswith("step index only")


def test_task_builder_fails_closed_on_group_overlap_and_sealed_audit():
    audit, split = _audit_and_split()
    audit["episodes"][1]["episode_metadata/recording_folderpath"] = "/recording/a"
    with pytest.raises(ValueError, match="recording-group overlap"):
        build_early_outcome_rows(audit, split)

    audit, split = _audit_and_split()
    split["sealed_audit"] = {"episode_ids": []}
    with pytest.raises(ValueError, match="sealed audit"):
        build_early_outcome_rows(audit, split)


def test_task_builder_reports_short_episode_exclusions_instead_of_silently_dropping():
    audit, split = _audit_and_split()
    audit["episodes"][0]["step_count"] = 28
    result = build_early_outcome_rows(audit, split)
    assert result["splits"]["train"]["episode_count"] == 0
    assert result["splits"]["train"]["excluded_short_episode_counts"] == {"success": 1}
