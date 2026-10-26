import hashlib

import pytest

from scripts.build_droid_group_split import build_group_split


def _episode(index: int, instruction: str | None = None):
    return {
        "episode_metadata/file_path": f"/source/{index}",
        "episode_metadata/recording_folderpath": f"/recording/{index}",
        "normalized_instructions": [instruction or f"task {index}"],
    }


def test_split_is_deterministic_and_excludes_debug_source_groups():
    audit = {
        "dataset_id": "droid/droid",
        "dataset_version": "1.4.0",
        "selection_manifest_sha256": "fixed-hash",
        "episodes": [_episode(index) for index in range(8)],
    }
    debug = {"episodes": [_episode(0)]}
    first = build_group_split(
        audit,
        debug,
        validation_groups=3,
        minimum_train_groups=3,
    )
    second = build_group_split(
        audit,
        debug,
        validation_groups=3,
        minimum_train_groups=3,
    )
    assert first == second
    assert first["excluded_debug_overlap_episodes"] == 1
    assert first["train"]["recording_groups"] >= 3
    assert first["validation"]["recording_groups"] >= 3
    assert not (set(first["train"]["episode_ids"]) & set(first["validation"]["episode_ids"]))


def test_split_unions_duplicate_instruction_content():
    rows = [_episode(index) for index in range(8)]
    rows[1]["normalized_instructions"] = ["same task"]
    rows[2]["normalized_instructions"] = ["same task"]
    audit = {
        "dataset_id": "droid/droid",
        "dataset_version": "1.4.0",
        "episodes": rows,
    }
    result = build_group_split(
        audit,
        {"episodes": []},
        validation_groups=3,
        minimum_train_groups=3,
    )
    validation_ids = set(result["validation"]["episode_ids"])
    repeated_ids = {
        hashlib.sha256(f"/source/{index}".encode()).hexdigest()
        for index in (1, 2)
    }
    assert repeated_ids <= validation_ids or repeated_ids.isdisjoint(validation_ids)


def test_split_fails_closed_when_group_floor_is_unavailable():
    audit = {
        "dataset_id": "droid/droid",
        "dataset_version": "1.4.0",
        "episodes": [_episode(index) for index in range(3)],
    }
    with pytest.raises(ValueError, match="need at least 5"):
        build_group_split(
            audit,
            {"episodes": []},
            validation_groups=3,
            minimum_train_groups=2,
        )
