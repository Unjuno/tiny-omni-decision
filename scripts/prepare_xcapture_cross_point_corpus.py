"""Freeze and acquire object-disjoint X-Capture train/validation cross-points.

The script downloads only the two seed-ranked complete points per train or
validation object. It never requests evaluation media. Each asset is checked
against the pinned ZIP central directory, uncompressed size, and CRC32.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import threading
import time
import zlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from scripts.audit_xcapture_sample import (
    ARCHIVE_URL,
    EXPECTED_ARCHIVE_SHA256,
    MODALITY_PATHS,
    REVISION,
    SOURCE_REPORT,
    SPLIT_MANIFEST,
    _central_directory,
    _parse_entries,
)

EXPECTED_CENTRAL_SHA256 = "3d8e546f1ca55e52a2b8e6c72aab99d14815e9da631589cb4d144f1179790f17"
REQUIRED_PATHS = {
    **MODALITY_PATHS,
    "object_mask": "point_cloud/mask.png",
}
SELECTION_RULE = "xcapture-cross-point-seed17-v1"
POINTS_PER_OBJECT = 2
DEFAULT_PLAN = Path(
    "C:/CodexArtifacts/pretrained-reuse-r2-local/"
    "xcapture-cross-point-plan-seed17-20261128.json"
)
_local = threading.local()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return sha256_bytes(encoded)


def _session() -> requests.Session:
    session = getattr(_local, "session", None)
    if session is None:
        session = requests.Session()
        retry = Retry(
            total=3,
            connect=3,
            read=3,
            status=3,
            backoff_factor=0.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET"}),
        )
        session.mount("https://", HTTPAdapter(max_retries=retry, pool_maxsize=2))
        _local.session = session
    return session


def _range(start: int, end: int) -> bytes:
    response = _session().get(
        ARCHIVE_URL,
        headers={"Range": f"bytes={start}-{end}"},
        timeout=(30, 90),
    )
    response.raise_for_status()
    expected_prefix = f"bytes {start}-{end}/"
    if response.status_code != 206 or not response.headers.get(
        "Content-Range", ""
    ).startswith(expected_prefix):
        raise RuntimeError("pinned archive server did not honor the byte range")
    payload = response.content
    if len(payload) != end - start + 1:
        raise RuntimeError("range response length does not match requested bytes")
    return payload


def choose_cross_points(
    entries: dict[str, dict[str, int]],
    split: dict[str, Any],
    *,
    points_per_object: int = POINTS_PER_OBJECT,
) -> list[dict[str, str]]:
    if points_per_object < 2:
        raise ValueError("cross-point comparisons require at least two points per object")
    allowed_partitions = ("train", "validation")
    split_ids = {
        partition: set(split["partitions"][partition])
        for partition in (*allowed_partitions, "evaluation")
    }
    if split_ids["train"] & split_ids["validation"]:
        raise ValueError("train and validation object IDs overlap")
    if (split_ids["train"] | split_ids["validation"]) & split_ids["evaluation"]:
        raise ValueError("development and evaluation object IDs overlap")

    expression = re.compile(r"^XCapture_data/(\d{5})/(\d{3})/(.+)$")
    required = set(REQUIRED_PATHS.values())
    complete: dict[tuple[str, str], set[str]] = defaultdict(set)
    for path in entries:
        match = expression.match(path)
        if match:
            object_id, point_id, suffix = match.groups()
            if suffix in required:
                complete[(object_id, point_id)].add(suffix)

    points_by_object: dict[str, list[str]] = defaultdict(list)
    for (object_id, point_id), assets in complete.items():
        if assets == required:
            points_by_object[object_id].append(point_id)

    result: list[dict[str, str]] = []
    for partition in allowed_partitions:
        for object_id in sorted(split_ids[partition]):
            candidates = sorted(points_by_object.get(object_id, []))
            if len(candidates) < points_per_object:
                raise ValueError(
                    f"{partition} object {object_id} has only {len(candidates)} "
                    f"complete points; need {points_per_object}"
                )
            ranked = sorted(
                candidates,
                key=lambda point_id: hashlib.sha256(
                    f"{SELECTION_RULE}|{partition}|{object_id}|{point_id}".encode()
                ).hexdigest(),
            )
            result.extend(
                {
                    "partition": partition,
                    "object_id": object_id,
                    "point_id": point_id,
                }
                for point_id in ranked[:points_per_object]
            )

    selected_ids = {row["object_id"] for row in result}
    if selected_ids & split_ids["evaluation"]:
        raise ValueError("evaluation object selected for media acquisition")
    return result


def build_plan(
    entries: dict[str, dict[str, int]],
    split: dict[str, Any],
    *,
    split_sha256: str,
    source_sha256: str,
    points_per_object: int = POINTS_PER_OBJECT,
) -> dict[str, Any]:
    if split.get("archive_sha256") != EXPECTED_ARCHIVE_SHA256:
        raise ValueError("split manifest does not match the pinned archive")
    points = choose_cross_points(entries, split, points_per_object=points_per_object)
    files = []
    for sample_id, point in enumerate(points, start=1):
        for modality, suffix in REQUIRED_PATHS.items():
            path = f"XCapture_data/{point['object_id']}/{point['point_id']}/{suffix}"
            if path not in entries:
                raise ValueError(f"selected asset is missing from archive index: {path}")
            files.append(
                {
                    "sample_id": sample_id,
                    **point,
                    "modality": modality,
                    "source_path": path,
                    **entries[path],
                }
            )
    point_counts = {
        partition: sum(row["partition"] == partition for row in points)
        for partition in ("train", "validation")
    }
    expected_points = {
        partition: len(split["partitions"][partition]) * points_per_object
        for partition in ("train", "validation")
    }
    if point_counts != expected_points:
        raise ValueError(f"selected point counts mismatch: {point_counts} != {expected_points}")
    plan = {
        "dataset_revision": REVISION,
        "archive_sha256": EXPECTED_ARCHIVE_SHA256,
        "central_directory_sha256": EXPECTED_CENTRAL_SHA256,
        "source_manifest_sha256": source_sha256,
        "split_manifest_sha256": split_sha256,
        "selection_rule": SELECTION_RULE,
        "points_per_object": points_per_object,
        "partition_object_counts": {
            partition: len(split["partitions"][partition])
            for partition in ("train", "validation")
        },
        "partition_point_counts": point_counts,
        "evaluation_object_count": len(split["partitions"]["evaluation"]),
        "evaluation_media_accessed": False,
        "selected_points": points,
        "files": files,
        "file_count": len(files),
        "compressed_bytes": sum(row["compressed_size"] for row in files),
        "uncompressed_bytes": sum(row["uncompressed_size"] for row in files),
    }
    plan["plan_sha256"] = canonical_hash(plan)
    return plan


def _verify_plan(plan: dict[str, Any]) -> None:
    supplied = plan.get("plan_sha256")
    payload = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if supplied != canonical_hash(payload):
        raise ValueError("plan hash mismatch")
    if plan.get("dataset_revision") != REVISION:
        raise ValueError("unexpected dataset revision")
    if plan.get("archive_sha256") != EXPECTED_ARCHIVE_SHA256:
        raise ValueError("unexpected archive hash")
    if plan.get("evaluation_media_accessed") is not False:
        raise ValueError("plan records evaluation media access")
    if any(row["partition"] not in {"train", "validation"} for row in plan["files"]):
        raise ValueError("plan contains a non-development file")


def _extract_entry(entry: dict[str, Any]) -> bytes:
    source_path = entry["source_path"]
    start = entry["local_header_offset"]
    local = _range(start, start + 29)
    fields = struct.unpack("<4s5H3I2H", local)
    if fields[0] != b"PK\x03\x04" or fields[3] != entry["method"]:
        raise RuntimeError(f"local ZIP header mismatch: {source_path}")
    file_name_size, extra_size = fields[9], fields[10]
    name_start = 30
    local_name = local[name_start : name_start + file_name_size]
    if len(local_name) != file_name_size:
        local = _range(start, start + 29 + file_name_size + extra_size)
        local_name = local[name_start : name_start + file_name_size]
    try:
        decoded_name = local_name.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RuntimeError(f"invalid local ZIP filename: {source_path}") from error
    if decoded_name != source_path:
        raise RuntimeError(f"local ZIP filename mismatch: {source_path}")
    payload_start = start + 30 + file_name_size + extra_size
    compressed_size = entry["compressed_size"]
    compressed = _range(payload_start, payload_start + compressed_size - 1)
    method = entry["method"]
    if method == 0:
        payload = compressed
    elif method == 8:
        payload = zlib.decompress(compressed, -15)
    else:
        raise RuntimeError(f"unsupported compression method {method}: {source_path}")
    if len(payload) != entry["uncompressed_size"]:
        raise RuntimeError(f"uncompressed size mismatch: {source_path}")
    if zlib.crc32(payload) & 0xFFFFFFFF != entry["crc32"]:
        raise RuntimeError(f"CRC mismatch: {source_path}")
    return payload


def download_plan(plan_path: Path, output_dir: Path, *, max_workers: int = 8) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite acquisition output: {output_dir}")
    if not 1 <= max_workers <= 12:
        raise ValueError("max_workers must be between 1 and 12")

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    _verify_plan(plan)
    source_path = SOURCE_REPORT
    split_path = SPLIT_MANIFEST
    source = json.loads(source_path.read_text(encoding="utf-8"))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if source.get("status") != "ALLOW_TASK_SCOPED_CROSS_SENSORY_RETRIEVAL":
        raise ValueError("X-Capture source-use gate is not ALLOW")
    if sha256_file(source_path) != plan["source_manifest_sha256"]:
        raise ValueError("source manifest changed since plan freeze")
    if sha256_file(split_path) != plan["split_manifest_sha256"]:
        raise ValueError("split manifest changed since plan freeze")

    central_session = requests.Session()
    directory, entry_count, _ = _central_directory(central_session)
    if entry_count != 94_973:
        raise ValueError("archive entry count changed")
    entries = _parse_entries(directory)
    current_plan = build_plan(
        entries,
        split,
        split_sha256=plan["split_manifest_sha256"],
        source_sha256=plan["source_manifest_sha256"],
        points_per_object=plan["points_per_object"],
    )
    if current_plan["plan_sha256"] != plan["plan_sha256"]:
        raise ValueError("frozen acquisition plan differs from current pinned archive index")

    required_free = plan["uncompressed_bytes"] + 2 * 1024**3
    free_bytes = shutil.disk_usage(output_dir.parent).free
    if free_bytes < required_free:
        raise OSError(
            f"insufficient free space: need {required_free} bytes including 2 GiB reserve; "
            f"have {free_bytes}"
        )
    output_dir.mkdir(parents=True)
    (output_dir / "acquisition-plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    started_at = datetime.now(UTC).isoformat()
    started = time.perf_counter()
    progress_path = output_dir / "download-progress.jsonl"
    completed: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    def fetch(item: dict[str, Any]) -> dict[str, Any]:
        payload = _extract_entry(item)
        extension = Path(item["source_path"]).suffix
        output_name = (
            f"sample-{item['sample_id']:04d}-{item['modality']}{extension}"
        )
        target = output_dir / output_name
        temporary = target.with_suffix(target.suffix + ".partial")
        temporary.write_bytes(payload)
        os.replace(temporary, target)
        return {
            **{
                key: item[key]
                for key in (
                    "sample_id",
                    "partition",
                    "object_id",
                    "point_id",
                    "modality",
                    "source_path",
                )
            },
            "output_path": output_name,
            "bytes": len(payload),
            "sha256": sha256_bytes(payload),
        }

    with progress_path.open("w", encoding="utf-8") as progress, ThreadPoolExecutor(
        max_workers=max_workers
    ) as pool:
        futures = {pool.submit(fetch, item): item for item in plan["files"]}
        for future in as_completed(futures):
            item = futures[future]
            try:
                row = future.result()
                completed.append(row)
                progress.write(json.dumps(row, sort_keys=True) + "\n")
                progress.flush()
            except Exception as error:  # Preserve a precise partial-run receipt.
                failures.append({"source_path": item["source_path"], "error": repr(error)})

    elapsed = time.perf_counter() - started
    report = {
        "status": "INCOMPLETE" if failures else "PASS_ACQUIRED_TRAIN_VALIDATION_ONLY",
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(UTC).isoformat(),
        "elapsed_seconds": elapsed,
        "dataset_revision": REVISION,
        "archive_sha256": EXPECTED_ARCHIVE_SHA256,
        "central_directory_sha256": EXPECTED_CENTRAL_SHA256,
        "plan_sha256": plan["plan_sha256"],
        "evaluation_media_accessed": False,
        "evaluation_objects_media_opened": 0,
        "media_policy": (
            f"{len(plan['selected_points'])} complete points from train/validation only; "
            "evaluation media unopened"
        ),
        "selected_point_count": len(plan["selected_points"]),
        "selected_file_count": len(plan["files"]),
        "sample_points": plan["selected_points"],
        "completed_file_count": len(completed),
        "completed_uncompressed_bytes": sum(row["bytes"] for row in completed),
        "partition_point_counts": plan["partition_point_counts"],
        "files": sorted(completed, key=lambda row: row["sample_id"]),
        "failures": failures,
    }
    receipt_path = output_dir / "sample-audit.json"
    receipt_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report["receipt_sha256"] = sha256_file(receipt_path)
    if failures:
        raise RuntimeError(
            f"acquisition incomplete ({len(completed)}/{len(plan['files'])}); "
            f"preserved partial data at {output_dir}"
        )
    return report


def _load_inputs(source_path: Path, split_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    source = json.loads(source_path.read_text(encoding="utf-8"))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if source.get("status") != "ALLOW_TASK_SCOPED_CROSS_SENSORY_RETRIEVAL":
        raise ValueError("source-use gate is not ALLOW")
    if split.get("archive_sha256") != EXPECTED_ARCHIVE_SHA256:
        raise ValueError("split manifest does not match pinned archive")
    return source, split


def make_plan(source_path: Path, split_path: Path, points_per_object: int) -> dict[str, Any]:
    source, split = _load_inputs(source_path, split_path)
    session = requests.Session()
    directory, entry_count, _ = _central_directory(session)
    if entry_count != 94_973:
        raise ValueError("archive entry count changed")
    entries = _parse_entries(directory)
    return build_plan(
        entries,
        split,
        split_sha256=sha256_file(split_path),
        source_sha256=sha256_file(source_path),
        points_per_object=points_per_object,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--make-plan", action="store_true")
    modes.add_argument("--execute-plan", type=Path)
    parser.add_argument("--source-report", type=Path, default=SOURCE_REPORT)
    parser.add_argument("--split-manifest", type=Path, default=SPLIT_MANIFEST)
    parser.add_argument("--plan-output", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--points-per-object", type=int, default=POINTS_PER_OBJECT)
    parser.add_argument("--max-workers", type=int, default=8)
    args = parser.parse_args()

    if args.make_plan:
        if args.plan_output.exists():
            raise FileExistsError(f"refusing to overwrite plan: {args.plan_output}")
        plan = make_plan(args.source_report, args.split_manifest, args.points_per_object)
        args.plan_output.parent.mkdir(parents=True, exist_ok=True)
        args.plan_output.write_text(
            json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "plan_path": str(args.plan_output),
                    "plan_sha256": plan["plan_sha256"],
                    "point_count": len(plan["selected_points"]),
                    "file_count": plan["file_count"],
                    "compressed_bytes": plan["compressed_bytes"],
                    "uncompressed_bytes": plan["uncompressed_bytes"],
                    "evaluation_media_accessed": False,
                },
                indent=2,
            )
        )
        return

    if args.output_dir is None:
        parser.error("--output-dir is required with --execute-plan")
    report = download_plan(args.execute_plan, args.output_dir, max_workers=args.max_workers)
    print(json.dumps({key: value for key, value in report.items() if key != "files"}, indent=2))


if __name__ == "__main__":
    main()
