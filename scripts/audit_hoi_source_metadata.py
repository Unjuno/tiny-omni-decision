"""Read-only inventory of Hoi! Dataverse file metadata (never downloads media)."""

from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from collections import defaultdict
from typing import Any

DATASET_VERSION = "1"
METADATA_URL = (
    "https://bonndata.uni-bonn.de/api/datasets/:persistentId/"
    "?persistentId=doi%3A10.60507%2FFK2%2FQODWTV"
)
SENSOR_FILES = {"force_torque.zip": "gripper", "telemetry.zip": "umi"}


def parse_environment_index_keys(files: list[dict[str, Any]], filename: str) -> set[str]:
    """Parse candidate environment/index keys from public package paths.

    Two-number suffixes denote inclusive ranges (e.g. ``1-14``); suffixes with
    more than two numbers are explicit index lists (e.g. ``1-3-5-7-9``). These
    are candidate keys, not proof of independent physical objects.
    """
    keys: set[str] = set()
    suffix_pattern = re.compile(r"^(?P<environment>.+)_(?P<indices>[0-9-]+)$")
    for entry in files:
        data_file = entry.get("dataFile", {})
        if data_file.get("filename") != filename:
            continue
        embodiment = SENSOR_FILES[filename]
        directory = str(entry.get("directoryLabel", ""))
        environment = directory.split("/", maxsplit=1)[0]
        leaf = directory.rsplit("/", maxsplit=1)[-1]
        tail = "_gripper_bag" if embodiment == "gripper" else "_umi"
        if not leaf.endswith(tail):
            raise ValueError(f"unexpected Hoi! package directory: {directory}")
        package_stem = leaf[: -len(tail)]
        match = suffix_pattern.fullmatch(package_stem)
        if match is None or match.group("environment") != environment:
            raise ValueError(f"cannot parse Hoi! package indices: {directory}")
        indices = [int(value) for value in match.group("indices").split("-")]
        if len(indices) == 2:
            if indices[1] < indices[0]:
                raise ValueError(f"descending Hoi! package range: {directory}")
            indices = list(range(indices[0], indices[1] + 1))
        keys.update(f"{environment}/{index}" for index in indices)
    return keys


def _sum_bytes(entries: list[dict[str, Any]]) -> int:
    return sum(int(entry["dataFile"]["filesize"]) for entry in entries)


def summarize_metadata_payload(payload: bytes) -> dict[str, Any]:
    """Hash and summarize one Dataverse JSON metadata response."""
    dataset = json.loads(payload)
    version = dataset["data"]["latestVersion"]
    files = list(version["files"])
    if str(version["versionNumber"]) != DATASET_VERSION:
        raise ValueError(f"unexpected Hoi! dataset version: {version['versionNumber']}")

    by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in files:
        by_name[str(entry["dataFile"]["filename"])].append(entry)

    telemetry_entries = by_name["telemetry.zip"]
    aria_rgb_bytes = 0
    aria_rgb_count = 0
    for entry in telemetry_entries:
        directory = str(entry["directoryLabel"])
        environment = directory.split("/", maxsplit=1)[0]
        leaf = directory.rsplit("/", maxsplit=1)[-1]
        if not leaf.startswith(f"{environment}_") or not leaf.endswith("_umi"):
            raise ValueError(f"unexpected UMI package directory: {directory}")
        session_stem = leaf[: -len("_umi")] + "_umi_vrs"
        paired_directory = f"{environment}/umi/aria_human/{session_stem}"
        matches = [
            file_entry
            for file_entry in by_name["camera_rgb.zip"]
            if file_entry.get("directoryLabel") == paired_directory
        ]
        if len(matches) > 1:
            raise ValueError(f"duplicate paired Hoi! RGB package: {paired_directory}")
        if matches:
            aria_rgb_count += 1
            aria_rgb_bytes += int(matches[0]["dataFile"]["filesize"])

    sensor_keys = {
        filename: parse_environment_index_keys(files, filename)
        for filename in SENSOR_FILES
    }
    interaction_json = [
        entry
        for entry in files
        if re.fullmatch(r"interaction_splitting_info.*\.json", entry["dataFile"]["filename"])
    ]
    roots = {str(entry.get("directoryLabel", "")).split("/", maxsplit=1)[0] for entry in files}

    selected_names = (
        "camera_rgb.zip",
        "force_torque.zip",
        "digit.zip",
        "telemetry.zip",
        "odometry.zip",
    )
    by_filename = {
        name: {"count": len(by_name[name]), "total_bytes": _sum_bytes(by_name[name])}
        for name in selected_names
    }
    force_keys = sensor_keys["force_torque.zip"]
    telemetry_keys = sensor_keys["telemetry.zip"]
    return {
        "dataset_version": str(version["versionNumber"]),
        "metadata_response_bytes": len(payload),
        "metadata_response_sha256": hashlib.sha256(payload).hexdigest(),
        "file_count": len(files),
        "total_bytes": _sum_bytes(files),
        "top_level_directory_count": len(roots),
        "interaction_window_json_count": len(interaction_json),
        "interaction_window_json_directory_count": len(
            {str(entry.get("directoryLabel", "")) for entry in interaction_json}
        ),
        "selected_file_types": by_filename,
        "candidate_environment_index_keys": {
            "force_torque": len(force_keys),
            "umi_telemetry": len(telemetry_keys),
            "union_deduplicated": len(force_keys | telemetry_keys),
            "interpretation": (
                "Filename-derived candidates only; not proof of physical-object independence."
            ),
        },
        "umi_telemetry_with_aria_human_rgb": {
            "paired_package_count": aria_rgb_count,
            "camera_rgb_total_bytes": aria_rgb_bytes,
            "telemetry_total_bytes": _sum_bytes(telemetry_entries),
        },
    }


def fetch_metadata() -> bytes:
    request = urllib.request.Request(
        METADATA_URL,
        headers={"User-Agent": "tiny-omni-decision-source-screen/1"},
    )
    with urllib.request.urlopen(request, timeout=45) as response:
        return response.read()


def main() -> None:
    print(json.dumps(summarize_metadata_payload(fetch_metadata()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
