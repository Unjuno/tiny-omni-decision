from __future__ import annotations

import pytest

from scripts.download_holoassist_selected_imu import select_imu_members


def fixture_inputs():
    sessions = [
        {"session_id": "train-a", "split": "train", "selection_sha256": "selection"},
        {
            "session_id": "validation-b",
            "split": "validation",
            "selection_sha256": "selection",
        },
    ]
    suffixes = (
        "Accelerometer_sync.txt",
        "Gyroscope_sync.txt",
        "Magnetometer_sync.txt",
    )
    members = []
    for session_index, row in enumerate(sessions):
        for sensor_index, suffix in enumerate(suffixes):
            members.append(
                {
                    "index": session_index * 3 + sensor_index,
                    "name": f"{row['session_id']}/Export_py/IMU/{suffix}",
                    "session_id": row["session_id"],
                    "size": 10 + sensor_index,
                    "offset_data": 1000 + session_index * 100 + sensor_index * 20,
                    "is_file": True,
                    "is_directory": False,
                }
            )
    return (
        {"files": sessions},
        {
            "payload_media_downloaded": False,
            "selection_sha256": "selection",
            "allowed_session_count": 2,
            "index": {"members": members},
        },
    )


def test_select_imu_members_requires_all_three_streams_for_every_split_session():
    media_manifest, archive_index = fixture_inputs()

    selected = select_imu_members(media_manifest, archive_index)

    assert len(selected) == 6
    assert {row["split"] for row in selected} == {"train", "validation"}
    assert {row["session_id"] for row in selected} == {"train-a", "validation-b"}
    assert all(row["size_bytes"] >= 10 for row in selected)
    assert all(row["archive_offset"] >= 1000 for row in selected)


def test_select_imu_members_rejects_missing_sensor_file():
    media_manifest, archive_index = fixture_inputs()
    archive_index["index"]["members"].pop()

    with pytest.raises(ValueError, match="exactly the three synchronized IMU files"):
        select_imu_members(media_manifest, archive_index)


def test_select_imu_members_rejects_unselected_session_and_duplicate_member():
    media_manifest, archive_index = fixture_inputs()
    foreign = dict(archive_index["index"]["members"][0])
    foreign.update(
        session_id="official-test",
        name="official-test/Export_py/IMU/Accelerometer_sync.txt",
    )
    archive_index["index"]["members"].append(foreign)

    with pytest.raises(ValueError, match="contains a session outside"):
        select_imu_members(media_manifest, archive_index)

    media_manifest, archive_index = fixture_inputs()
    archive_index["index"]["members"].append(dict(archive_index["index"]["members"][0]))
    with pytest.raises(ValueError, match="duplicate archive member"):
        select_imu_members(media_manifest, archive_index)
