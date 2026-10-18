from __future__ import annotations

import pytest

from scripts.select_holoassist_companion_media import build_companion_plan


def fixture() -> tuple[dict, dict, set[str]]:
    base = {"selection": {}}
    index_members = []
    allowed = set()
    for split, count, prefix in (("train", 160, "tr"), ("validation", 207, "va")):
        videos = []
        for number in range(count):
            session_id = f"{prefix}-{number:03d}"
            allowed.add(session_id)
            videos.append({"session_id": session_id, "task_type": "assemble"})
            index_members.append(
                {
                    "session_id": session_id,
                    "name": f"{session_id}/Export_py/Video_pitchshift.mp4",
                    "offset_data": number * 1000,
                    "size": number + 100,
                    "is_file": True,
                }
            )
        base["selection"][split] = {"videos": videos}
    index = {
        "payload_media_downloaded": False,
        "archive_url": "https://example.invalid/video_pitch_shifted.tar",
        "archive_size_bytes": 123456,
        "archive_etag": '"stable"',
        "index": {"members": index_members},
    }
    return base, index, allowed


def test_companion_plan_preserves_fixed_split_order_and_replaces_member_paths() -> None:
    base, index, allowed = fixture()

    plan = build_companion_plan(base, index, allowed)

    assert plan["train"]["session_count"] == 160
    assert plan["validation"]["session_count"] == 207
    assert plan["train"]["videos"][0]["session_id"] == "tr-000"
    assert plan["train"]["videos"][0]["video_member_name"].endswith(
        "/Export_py/Video_pitchshift.mp4"
    )
    assert plan["all_selected_video_bytes"] == sum(
        row["size_bytes"]
        for split in ("train", "validation")
        for row in plan[split]["videos"]
    )


def test_companion_plan_rejects_missing_media_or_changed_split_size() -> None:
    base, index, allowed = fixture()
    index["index"]["members"].pop()
    with pytest.raises(ValueError, match="official train/validation allowlist"):
        build_companion_plan(base, index, allowed)

    base, index, allowed = fixture()
    base["selection"]["train"]["videos"].pop()
    with pytest.raises(ValueError, match="fixed 160/207 pilot"):
        build_companion_plan(base, index, allowed)


def test_companion_plan_rejects_duplicate_or_unverified_members() -> None:
    base, index, allowed = fixture()
    index["index"]["members"].append(index["index"]["members"][0])
    with pytest.raises(ValueError, match="duplicate pitch-shifted"):
        build_companion_plan(base, index, allowed)

    base, index, allowed = fixture()
    index["payload_media_downloaded"] = True
    with pytest.raises(ValueError, match="header-only"):
        build_companion_plan(base, index, allowed)


def test_companion_plan_rejects_index_session_outside_allowlist() -> None:
    base, index, allowed = fixture()
    allowed.add("test-session")
    with pytest.raises(ValueError, match="allowlist"):
        build_companion_plan(base, index, allowed)
