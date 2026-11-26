from __future__ import annotations

import pytest

from scripts.build_physicalai_singlearm_split import build_split_manifest


def _world_assets(count: int) -> list[dict[str, object]]:
    return [
        {
            "rfilename": (
                "panda-open-cabinet-left/videos/"
                f"chunk-{index // 1000:03d}/"
                f"observation.images.world_camera/episode_{index:06d}.mp4"
            ),
            "size": 1000 + index,
            "lfs": {"sha256": f"{index + 1:064x}"},
        }
        for index in range(count)
    ]


def test_split_is_deterministic_complete_and_episode_grouped():
    first = build_split_manifest(_world_assets(1512), total_episodes=1512, seed=17)
    second = build_split_manifest(_world_assets(1512), total_episodes=1512, seed=17)

    assert first == second
    assert first["split_counts"] == {
        "train": 1058,
        "validation": 227,
        "reserved_holdout": 227,
    }
    rows = first["episodes"]
    assert len(rows) == 1512
    assert [row["episode_index"] for row in rows] == list(range(1512))
    assert len({row["video_sha256"] for row in rows}) == 1512
    assert set(first["split_hashes"]) == {"train", "validation", "reserved_holdout"}
    assert len(first["pilot_selection"]["train_episode_indices"]) == 64
    assert len(first["pilot_selection"]["validation_episode_indices"]) == 16
    assert not (
        set(first["pilot_selection"]["train_episode_indices"])
        & set(first["pilot_selection"]["validation_episode_indices"])
    )


def test_split_rejects_missing_episode_video_asset():
    with pytest.raises(ValueError, match="exactly one asset"):
        build_split_manifest(_world_assets(9), total_episodes=10, seed=17)


def test_split_rejects_duplicate_episode_identity_or_media_hash():
    assets = _world_assets(2)
    assets.append(assets[0])
    with pytest.raises(ValueError, match="duplicate episode"):
        build_split_manifest(assets, total_episodes=2, seed=17)

    duplicate_media = _world_assets(2)
    duplicate_media[1]["lfs"] = duplicate_media[0]["lfs"]
    with pytest.raises(ValueError, match="duplicate world-camera media hash"):
        build_split_manifest(duplicate_media, total_episodes=2, seed=17)


def test_split_ignores_non_world_camera_files_but_fails_on_bad_world_hash():
    assets = _world_assets(2)
    assets.append(
        {
            "rfilename": "panda-open-cabinet-left/meta/info.json",
            "size": 4,
            "blobId": "deadbeef",
        }
    )
    assert len(build_split_manifest(assets, total_episodes=2, seed=17)["episodes"]) == 2

    assets[0]["lfs"] = {"sha256": "not-a-sha256"}
    with pytest.raises(ValueError, match="valid SHA-256"):
        build_split_manifest(assets, total_episodes=2, seed=17)
