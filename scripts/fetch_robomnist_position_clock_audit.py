"""Fetch one previously observed RoboMNIST TRAIN PositionGT file for a clock audit."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
import zipfile
import zlib
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

from scripts.audit_robomnist_source import sha
from scripts.fetch_starss23_smoke import HttpRangeReader

ROOT = Path("C:/CodexArtifacts/pretrained-reuse-r2")
GROUP = "Ar1_Vhigh_Ac0_Rp01_UNC062"
MEMBER = "00_Ar1_Vhigh_Ac0/Rp01_Ac0_Vhigh_Ar1_UNC062_Rxall_pos.json"
PARENT = ROOT / "robomnist-all-directories-replay-02.json"
PARENT_SHA = "b34bc54f57d3c434947d673ed32cd9a9c6e4f82bfc10f9b02213b397a76a29a4"
DIRECTORY = ROOT / "robomnist-all-directories-01/directory-00.json"
DIRECTORY_SHA = "900da2375691904f75477ec312f91c8d60cf9c3fb1eb12f861d57a8228694c08"
GROUP_RECEIPT = ROOT / "robomnist-source-audit-01/result-verified-02.groups.json"
GROUP_RECEIPT_SHA = "3476b1ce545e62e39b94a86bec70d3d6f1eb2ad21c27572d3fedc2b7089b5fce"
SOURCE_REPORT = ROOT / "robomnist-source-audit-01/result-verified-02.json"
SOURCE_REPORT_SHA = "e0bcd4a97876644059d24c96aa2381146c886ad1bbe1e0fcde330dd7991f4a59"
PREVIOUS_INPUT = ROOT / "robomnist-input-smoke-01/result.json"
PREVIOUS_INPUT_SHA = "5936b74193b1fdf8f813d85a5eb2391c48123a3eeb671153a0aad89f1980227b"
PREVIOUS_CSI = ROOT / "robomnist-input-smoke-01/csi.json"
PREVIOUS_CSI_SHA = "b70b8da68da71c4d7019d539b03fbdff04c851f89a2b6988940560bd0e6658c7"
PROTOCOL = Path("docs/pretrained_reuse/R2_ROBOMNIST_POSITION_CLOCK_PROTOCOL_20261102.md")
PROTOCOL_SHA = "477c1d9f759c56856fceef0c2f9d58264d1376a75274a8876b283493ef9e8445"
TRANSFER_LIMIT = 1 * 2**20
BODY_SECONDS = 120
OUTPUT_LIMIT = 1 * 2**20
DISK_RESERVE = 1 * 2**30


def extract_timestamp_summaries(value: object) -> list[dict[str, object]]:
    """Summarize timestamp-like fields without returning position/label values."""
    summaries: list[dict[str, object]] = []
    time_key = re.compile(r"timestamp|time_stamp|(^|_)time($|_)|(^|_)pts($|_)|clock|epoch", re.I)

    def numeric_summary(path: str, values: list[object]) -> None:
        try:
            numbers = [float(item) for item in values]
        except (TypeError, ValueError):
            summaries.append(
                {
                    "path": path,
                    "kind": "text",
                    "count": len(values),
                    "minimum": None,
                    "maximum": None,
                    "strictly_increasing": None,
                }
            )
            return
        finite = all(map(__import__("math").isfinite, numbers))
        summaries.append(
            {
                "path": path,
                "kind": "numeric" if finite else "nonfinite",
                "count": len(numbers),
                "minimum": min(numbers) if numbers and finite else None,
                "maximum": max(numbers) if numbers and finite else None,
                "strictly_increasing": (
                    all(a < b for a, b in zip(numbers, numbers[1:], strict=False))
                    if finite and len(numbers) > 1
                    else None
                ),
            }
        )

    def visit(
        node: object, path: str = "", skip_keys: frozenset[str] = frozenset()
    ) -> None:
        if isinstance(node, Mapping):
            for key, item in node.items():
                key_text = str(key)
                if key_text in skip_keys:
                    continue
                child = f"{path}.{key_text}" if path else key_text
                if time_key.search(key_text):
                    if isinstance(item, (list, tuple)):
                        if item and all(
                            isinstance(v, (int, float)) and not isinstance(v, bool)
                            for v in item
                        ):
                            numeric_summary(child, list(item))
                        elif item and all(isinstance(v, Mapping) for v in item):
                            visit(item, child)
                        elif not item:
                            summaries.append(
                                {
                                    "path": child,
                                    "kind": "empty",
                                    "count": 0,
                                    "minimum": None,
                                    "maximum": None,
                                    "strictly_increasing": None,
                                }
                            )
                        else:
                            numeric_summary(child, list(item))
                    else:
                        numeric_summary(child, [item])
                elif isinstance(item, (Mapping, list, tuple)):
                    visit(item, child)
        elif isinstance(node, (list, tuple)):
            record_keys = (
                set.intersection(*(set(item) for item in node))
                if node and all(isinstance(item, Mapping) for item in node)
                else set()
            )
            timestamp_keys = frozenset(
                str(key) for key in record_keys if time_key.search(str(key))
            )
            for key in sorted(timestamp_keys):
                values = [item[key] for item in node if isinstance(item, Mapping)]
                if values and all(
                    isinstance(item, (int, float)) and not isinstance(item, bool)
                    for item in values
                ):
                    numeric_summary(f"{path}[].{key}" if path else f"[].{key}", values)
            for item in node:
                if isinstance(item, Mapping):
                    visit(
                        item,
                        f"{path}[]" if path else "[]",
                        skip_keys=timestamp_keys,
                    )

    visit(value)
    return summaries


def peak_working_set_bytes() -> int:
    if os.name != "nt":
        raise RuntimeError("working-set guard is Windows-only")
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
            (key, ctypes.c_size_t)
            for key in [
                "PeakWorkingSetSize",
                "WorkingSetSize",
                "QuotaPeakPagedPoolUsage",
                "QuotaPagedPoolUsage",
                "QuotaPeakNonPagedPoolUsage",
                "QuotaNonPagedPoolUsage",
                "PagefileUsage",
                "PeakPagefileUsage",
            ]
        ]

    kernel = ctypes.WinDLL("kernel32")
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    api = ctypes.WinDLL("psapi").GetProcessMemoryInfo
    api.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    value = Counters()
    value.cb = ctypes.sizeof(value)
    if not api(kernel.GetCurrentProcess(), ctypes.byref(value), value.cb):
        raise OSError("process peak counter unavailable")
    return int(value.PeakWorkingSetSize)


def verify_inputs() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    expected = {
        PARENT: PARENT_SHA,
        DIRECTORY: DIRECTORY_SHA,
        GROUP_RECEIPT: GROUP_RECEIPT_SHA,
        SOURCE_REPORT: SOURCE_REPORT_SHA,
        PREVIOUS_INPUT: PREVIOUS_INPUT_SHA,
        PREVIOUS_CSI: PREVIOUS_CSI_SHA,
        PROTOCOL: PROTOCOL_SHA or sha(PROTOCOL),
    }
    if any(sha(path) != digest for path, digest in expected.items()):
        raise ValueError("pinned parent source/protocol artifact hash mismatch")
    parent = json.loads(PARENT.read_bytes())
    directory = json.loads(DIRECTORY.read_bytes())
    groups = json.loads(GROUP_RECEIPT.read_bytes())
    source = json.loads(SOURCE_REPORT.read_bytes())
    previous = json.loads(PREVIOUS_INPUT.read_bytes())
    if (
        parent["status"] != "PASS_COMPLETE_DIRECTORY_GROUP_ACCOUNTING_ONLY"
        or parent["archives"] != 60
        or parent["sealed_audit_opened"]
        or source["upstream_commit"] != "70f5e6393acf6128a7ceb7afacf4463060626ca3"
        or source["media_members_opened"] is not False
        or source["sealed_audit_opened"]
        or previous["status"] != "PASS_FIXED_NATIVE_INPUT_MECHANICS_ONLY"
        or previous["position_gt_opened"]
        or previous["sealed_audit_opened"]
        or previous["config"]["group"]["group_id"] != GROUP
    ):
        raise ValueError("known training-side, non-audit source receipts required")
    matched = [row for row in groups if row["group_id"] == GROUP]
    if len(matched) != 1 or matched[0]["supervision_only_member"] != MEMBER:
        raise ValueError("fixed supervision-only member/group mismatch")
    archive = directory["source_archive"]
    info = next((row for row in directory["members"] if row["name"] == MEMBER), None)
    if (
        archive["name"] != "00_Ar1_Vhigh_Ac0.zip"
        or archive["size"] != 889_834_555
        or archive["id"] != 51_588_737
        or archive["supplied_md5"] != "99a61bc10010855189daa45d0ad3b968"
        or info is None
        or info["directory"]
        or info["size"] != 313_307
        or info["compressed_size"] != 31_956
        or info["crc32"] != "082e8e4d"
        or info["compression"] != 8
        or info["offset"] != 888_233_944
        or PurePosixPath(MEMBER).name != "Rp01_Ac0_Vhigh_Ar1_UNC062_Rxall_pos.json"
    ):
        raise ValueError("fixed bounded source archive/member metadata mismatch")
    return parent, directory, info


class BoundedInputReader(HttpRangeReader):
    def __init__(self, archive: dict[str, object], output: Path, started: float):
        super().__init__(
            str(archive["download_url"]),
            int(archive["size"]),
            budget=TRANSFER_LIMIT,
            retries=0,
        )
        self.output = output
        self.started = started
        self.request_samples: list[dict[str, float | int]] = []

    def read(self, size: int = -1) -> bytes:
        if time.perf_counter() - self.started > BODY_SECONDS:
            raise TimeoutError("PositionGT clock audit exceeded body time limit")
        if shutil.disk_usage(self.output).free < DISK_RESERVE:
            raise OSError("disk free space fell below the 1GiB reserve")
        request_started = time.perf_counter()
        data = super().read(size)
        elapsed = time.perf_counter() - request_started
        sample = {"bytes": len(data), "elapsed_seconds": elapsed}
        self.request_samples.append(sample)
        print(json.dumps({"range_sample": len(self.request_samples), **sample}), flush=True)
        return data


def run(output: Path) -> dict[str, object]:
    _, directory, info_record = verify_inputs()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started_wall = datetime.now(UTC).isoformat()
    started = time.perf_counter()
    archive_record = directory["source_archive"]
    runner_sha = sha(Path(__file__))
    config = {
        "protocol": str(PROTOCOL),
        "protocol_sha256": sha(PROTOCOL),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_group": GROUP,
        "source_member": MEMBER,
        "supervision_only": True,
        "model_or_label_values_used": False,
        "sealed_audit_opened": False,
        "transfer_cap_bytes": TRANSFER_LIMIT,
        "body_seconds": BODY_SECONDS,
        "output_cap_bytes": OUTPUT_LIMIT,
        "disk_reserve_bytes": DISK_RESERVE,
        "source_archive": archive_record,
        "source_member_metadata": info_record,
        "runner_sha256": runner_sha,
        "parent_hashes": {
            str(path): digest
            for path, digest in {
                PARENT: PARENT_SHA,
                DIRECTORY: DIRECTORY_SHA,
                GROUP_RECEIPT: GROUP_RECEIPT_SHA,
                SOURCE_REPORT: SOURCE_REPORT_SHA,
                PREVIOUS_INPUT: PREVIOUS_INPUT_SHA,
                PREVIOUS_CSI: PREVIOUS_CSI_SHA,
            }.items()
        },
        "previous_input_sync_status": "UNVERIFIED_COMMON_CLOCK",
    }
    (output / "effective-config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    reader = BoundedInputReader(archive_record, output, started)
    target = None
    with zipfile.ZipFile(reader) as archive:
        actual = archive.getinfo(MEMBER)
        if (
            actual.file_size != info_record["size"]
            or actual.compress_size != info_record["compressed_size"]
            or f"{actual.CRC:08x}" != info_record["crc32"]
            or actual.compress_type != zipfile.ZIP_DEFLATED
            or actual.header_offset != info_record["offset"]
        ):
            raise ValueError("live ZIP central/local member metadata differs from pinned directory")
        target = archive.read(actual)  # Reads only this member; ZipExtFile verifies CRC.
    if (
        len(target) != info_record["size"]
        or zlib.crc32(target) & 0xFFFFFFFF != int(info_record["crc32"], 16)
    ):
        raise ValueError("PositionGT extracted size or CRC mismatch")
    if len(target) > OUTPUT_LIMIT:
        raise ValueError("PositionGT output exceeded 1MiB cap")
    parsed = json.loads(target)
    timestamp_summaries = extract_timestamp_summaries(parsed)
    result = {
        "status": "PASS_EXTRACTION_METADATA_AUDIT_PENDING_CLOCK_DECISION",
        "group_id": GROUP,
        "member": MEMBER,
        "training_side_known_group": True,
        "position_gt_role": "supervision-only; timestamp fields inspected only",
        "position_or_label_values_reported": False,
        "raw_position_gt_saved": False,
        "timestamp_summaries": timestamp_summaries,
        "clock_related_timestamp_field_names_present": any(
            "clock" in str(row["path"]).lower() or "epoch" in str(row["path"]).lower()
            for row in timestamp_summaries
        ),
        "clock_bridge_decision": "UNVERIFIED_PENDING_SOURCE_METADATA_REVIEW",
        "existing_video_timing": [
            {
                "receiver": row["receiver"],
                "time_base": row["time_base"],
                "first_frame_time_seconds": row["first_frame_time_seconds"],
                "last_frame_time_seconds": row["last_frame_time_seconds"],
                "median_interval_seconds": row["median_interval_seconds"],
            }
            for row in json.loads(PREVIOUS_INPUT.read_bytes())["video"]
        ],
        "existing_csi_timing": json.loads(PREVIOUS_INPUT.read_bytes())["csi"],
        "transfer_bytes": reader.transferred,
        "range_request_samples": reader.request_samples,
        "mean_range_request_seconds": (
            sum(sample["elapsed_seconds"] for sample in reader.request_samples)
            / len(reader.request_samples)
            if reader.request_samples
            else None
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_working_set_bytes": peak_working_set_bytes(),
        "started_utc": started_wall,
        "runner_sha256": runner_sha,
        "config_sha256": sha(output / "effective-config.json"),
        "position_file_bytes": len(target),
        "position_file_crc32": f"{zlib.crc32(target) & 0xFFFFFFFF:08x}",
        "position_file_sha256": hashlib.sha256(target).hexdigest(),
        "sealed_audit_opened": False,
        "training": False,
        "model_inference": False,
        "gpu_used": False,
        "cloud_cost": 0,
    }
    (output / "clock-audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: result[key]
                for key in [
                    "status",
                    "timestamp_summaries",
                    "transfer_bytes",
                    "range_request_samples",
                    "mean_range_request_seconds",
                    "elapsed_seconds",
                    "position_file_sha256",
                ]
            },
            indent=2,
        )
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
