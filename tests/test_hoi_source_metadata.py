from __future__ import annotations

import json

import pytest

from scripts.audit_hoi_source_metadata import (
    parse_environment_index_keys,
    summarize_metadata_payload,
)


def file_entry(directory: str, filename: str, size: int) -> dict[str, object]:
    return {
        "directoryLabel": directory,
        "dataFile": {"filename": filename, "filesize": size},
    }


def test_parse_environment_index_keys_ranges_and_explicit_lists() -> None:
    files = [
        file_entry("bathroom_1/gripper/gripper/bathroom_1_1-2_gripper_bag", "force_torque.zip", 10),
        file_entry("kitchen_7/umi/umi_gripper/kitchen_7_1-3-5-7-9_umi", "telemetry.zip", 20),
    ]

    assert parse_environment_index_keys(files, "force_torque.zip") == {
        "bathroom_1/1",
        "bathroom_1/2",
    }
    assert parse_environment_index_keys(files, "telemetry.zip") == {
        "kitchen_7/1",
        "kitchen_7/3",
        "kitchen_7/5",
        "kitchen_7/7",
        "kitchen_7/9",
    }


def test_summarize_metadata_payload_reports_union_and_paired_rgb() -> None:
    document = {
        "data": {
            "latestVersion": {
                "versionNumber": 1,
                "files": [
                    file_entry(
                        "bathroom_1/gripper/gripper/bathroom_1_1-2_gripper_bag",
                        "force_torque.zip",
                        10,
                    ),
                    file_entry(
                        "bathroom_1/umi/umi_gripper/bathroom_1_1-2_umi",
                        "telemetry.zip",
                        5,
                    ),
                    file_entry(
                        "bathroom_1/umi/aria_human/bathroom_1_1-2_umi_vrs",
                        "camera_rgb.zip",
                        100,
                    ),
                    file_entry(
                        "bathroom_1/gripper",
                        "interaction_splitting_info_1-2_confirmed.json",
                        7,
                    ),
                ],
            }
        }
    }

    result = summarize_metadata_payload(json.dumps(document).encode())

    assert result["file_count"] == 4
    assert result["candidate_environment_index_keys"] == {
        "force_torque": 2,
        "umi_telemetry": 2,
        "union_deduplicated": 2,
        "interpretation": (
            "Filename-derived candidates only; not proof of physical-object independence."
        ),
    }
    assert result["umi_telemetry_with_aria_human_rgb"] == {
        "paired_package_count": 1,
        "camera_rgb_total_bytes": 100,
        "telemetry_total_bytes": 5,
    }
    assert len(result["metadata_response_sha256"]) == 64


def test_summarize_rejects_unexpected_dataset_version() -> None:
    payload = json.dumps({"data": {"latestVersion": {"versionNumber": 2, "files": []}}}).encode()

    with pytest.raises(ValueError, match="unexpected Hoi! dataset version"):
        summarize_metadata_payload(payload)
