from __future__ import annotations

import zipfile

import pytest

from scripts.preflight_perception_test_mcqa_media import plan_split_members


def test_plans_only_requested_video_members_in_stable_order(tmp_path):
    archive_path = tmp_path / "valid_videos.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("videos/video_2.mp4", b"second video payload")
        archive.writestr("videos/video_1.mp4", b"first video")
        archive.writestr("videos/video_999.mp4", b"unselected")

    with zipfile.ZipFile(archive_path) as archive:
        plan = plan_split_members("valid", {"video_2", "video_1"}, archive)

    assert plan["video_groups"] == 2
    assert [row["video_id"] for row in plan["members"]] == ["video_1", "video_2"]
    assert plan["expanded_bytes"] == len(b"first video") + len(b"second video payload")
    assert plan["compressed_bytes"] > 0
    assert all(row["member"].startswith("videos/") for row in plan["members"])


def test_plan_fails_closed_for_missing_or_test_split_members(tmp_path):
    archive_path = tmp_path / "valid_videos.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("videos/video_1.mp4", b"video")

    with zipfile.ZipFile(archive_path) as archive:
        with pytest.raises(ValueError, match="missing"):
            plan_split_members("valid", {"video_2"}, archive)
        with pytest.raises(ValueError, match="only official train"):
            plan_split_members("test", {"video_1"}, archive)
