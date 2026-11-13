from __future__ import annotations

import hashlib

import pytest

from scripts.build_physicalai_stack_platforms_split import build_manifest


def _world_videos(count: int) -> list[dict[str, object]]:
    result = []
    for episode in range(count):
        digest = hashlib.sha256(f"world-{episode}".encode()).hexdigest()
        result.append(
            {
                "path": (
                    "panda-stack-platforms/videos/chunk-000/"
                    f"observation.images.world_camera/episode_{episode:06d}.mp4"
                ),
                "size": 100 + episode,
                "sha256": digest,
            }
        )
    return result


def _episode_data(count: int) -> list[dict[str, object]]:
    result = []
    for episode in range(count):
        digest = hashlib.sha256(f"parquet-{episode}".encode()).hexdigest()
        result.append(
            {
                "path": (
                    "panda-stack-platforms/data/chunk-000/"
                    f"episode_{episode:06d}.parquet"
                ),
                "size": 50 + episode,
                "sha256": digest,
            }
        )
    return result


def test_episode_split_is_deterministic_disjoint_and_selects_only_project_groups():
    rows = _world_videos(20)
    data = _episode_data(20)

    first = build_manifest(
        rows,
        data,
        total_episodes=20,
        seed=17,
        pilot_train_groups=4,
        pilot_validation_groups=3,
        full_subset_bytes=100_000,
    )
    second = build_manifest(
        rows,
        data,
        total_episodes=20,
        seed=17,
        pilot_train_groups=4,
        pilot_validation_groups=3,
        full_subset_bytes=100_000,
    )

    assert first["split_hashes"] == second["split_hashes"]
    assert first["split_counts"] == {
        "train": 14,
        "validation": 3,
        "reserved_holdout": 3,
    }
    selection = first["pilot_selection"]
    assert len(selection["train_episode_indices"]) == 4
    assert len(selection["validation_episode_indices"]) == 3
    assert set(selection["train_episode_indices"]).isdisjoint(
        selection["validation_episode_indices"]
    )
    assignments = {row["episode_index"]: row["split"] for row in first["episode_assignments"]}
    assert all(assignments[index] == "train" for index in selection["train_episode_indices"])
    assert all(
        assignments[index] == "validation" for index in selection["validation_episode_indices"]
    )
    assert first["grouping"]["cross_split_exact_world_video_sha256_overlap"] == 0
    assert first["split_policy"]["reserved_payload_loaded"] is False
    assert selection["selected_media_payload_bytes"] == sum(
        rows[index]["size"] + data[index]["size"]
        for index in (
            *selection["train_episode_indices"],
            *selection["validation_episode_indices"],
        )
    )


def test_split_fails_closed_on_missing_group_or_repeated_media_hash():
    with pytest.raises(ValueError, match="missing world-camera asset"):
        build_manifest(_world_videos(4)[:-1], _episode_data(4), total_episodes=4)

    rows = _world_videos(4)
    rows[1]["sha256"] = rows[0]["sha256"]
    with pytest.raises(ValueError, match="duplicate world-camera SHA-256"):
        build_manifest(rows, _episode_data(4), total_episodes=4)

    data = _episode_data(4)
    data[1]["sha256"] = data[0]["sha256"]
    with pytest.raises(ValueError, match="duplicate episode-Parquet SHA-256"):
        build_manifest(_world_videos(4), data, total_episodes=4)


def test_split_fails_closed_without_lfs_hash():
    rows = _world_videos(4)
    rows[0]["sha256"] = None
    with pytest.raises(ValueError, match="requires LFS SHA-256"):
        build_manifest(rows, _episode_data(4), total_episodes=4)


def test_split_fails_closed_when_parquet_identity_does_not_match_video():
    data = _episode_data(4)[1:]
    with pytest.raises(ValueError, match="missing episode Parquet asset"):
        build_manifest(_world_videos(4), data, total_episodes=4)
