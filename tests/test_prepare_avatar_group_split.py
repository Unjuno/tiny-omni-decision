from __future__ import annotations

import zipfile

import pytest

from scripts.prepare_avatar_group_split import build_manifest


def _archive(path, names):
    with zipfile.ZipFile(path, "w") as archive:
        for name in names:
            archive.writestr(name, b"media bytes are never read by the split builder")
    return path


def test_split_groups_multiple_clips_by_source_video_id_and_ignores_sidecars(tmp_path):
    names = [f"video/{i:011d}_{i:05d}.mp4" for i in range(30)]
    names += ["video/00000000001_99999.mp4", "__MACOSX/video/._00000000001_00001.mp4"]
    result = build_manifest(_archive(tmp_path / "video.zip", names), check_pin=False)

    assert result["video_clip_count"] == 31
    assert result["source_video_group_count"] == 30
    assert result["groups_with_multiple_clips"] == 1
    assert result["ignored_mac_osx_sidecars"] == 1
    splits = result["splits"]
    assert {name: split["source_video_group_count"] for name, split in splits.items()} == {
        "train": 24,
        "validation": 3,
        "audit": 3,
    }
    group_sets = [set(split["source_video_ids"]) for split in splits.values()]
    assert not any(
        group_sets[left] & group_sets[right]
        for left, right in ((0, 1), (0, 2), (1, 2))
    )
    assert sum(split["video_clip_count"] for split in splits.values()) == 31


def test_split_is_deterministic_for_same_group_ids(tmp_path):
    names = [f"video/{i:011d}_{i:05d}.mp4" for i in range(30)]
    first = build_manifest(_archive(tmp_path / "first.zip", names), check_pin=False)
    second = build_manifest(_archive(tmp_path / "second.zip", reversed(names)), check_pin=False)

    assert first["splits"] == second["splits"]


def test_split_fails_closed_on_unrecognized_video_member(tmp_path):
    path = _archive(tmp_path / "video.zip", ["video/not-a-source-id.mp4"])

    with pytest.raises(ValueError, match="unrecognized AVATAR video member"):
        build_manifest(path, check_pin=False)
