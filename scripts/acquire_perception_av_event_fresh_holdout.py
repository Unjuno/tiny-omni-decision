"""Acquire a pinned, TRAIN-only fresh video holdout without loading a model."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
import zipfile
from pathlib import Path

from scripts.audit_mmfit import RemoteZip
from scripts.audit_perception_av_correspondence import sha256_file
from scripts.plan_perception_av_event_fresh_holdout import (
    MAX_CLIP_BYTES,
    MAX_SPLIT_BYTES,
    PLAN_SHA256,
    ROOT,
    TRAIN_ARCHIVE_PIN,
    TRAIN_MEDIA_MANIFEST_SHA256,
    VALID_MEDIA_MANIFEST_SHA256,
)
from scripts.plan_perception_av_event_media import MEDIA_MANIFESTS

RESERVE_BYTES = 8_000_000_000
CHUNK_BYTES = 1024 * 1024


def write_receipt(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def validate_plan(plan_path: Path, expected_sha256: str) -> dict:
    if sha256_file(plan_path) != expected_sha256:
        raise ValueError("fresh holdout plan hash mismatch")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    rows = plan.get("rows")
    if (
        plan.get("status") != "PASS_FRESH_TRAIN_HOLDOUT_PLAN"
        or plan.get("official_split", "").split(";")[0] != "TRAIN only"
        or plan.get("training") is not False
        or plan.get("checkpoint_selection") is not False
        or plan.get("test_opened") is not False
        or plan.get("archive_pin") != TRAIN_ARCHIVE_PIN
        or plan.get("source_revision") != "3938d2f1ba3a6b502025741cea4cd73c7b3bdfaf"
        or not isinstance(rows, list)
        or len(rows) != 200
    ):
        raise ValueError("plan is not the fixed, inference-only 200-video holdout")
    ids = [row.get("video_id") for row in rows]
    if len(set(ids)) != len(ids) or any(not isinstance(video_id, str) for video_id in ids):
        raise ValueError("holdout contains duplicate or invalid video IDs")
    if any(
        row.get("metadata", {}).get("split") != "train"
        or row.get("metadata", {}).get("video_id") != row.get("video_id")
        or row.get("member") != f"videos/{row.get('video_id')}.mp4"
        or not 0 < row.get("bytes", 0) <= MAX_CLIP_BYTES
        for row in rows
    ):
        raise ValueError("invalid TRAIN identity, source member or clip size")
    total = sum(row["bytes"] for row in rows)
    if total != plan.get("total_uncompressed_bytes") or total > MAX_SPLIT_BYTES:
        raise ValueError("holdout media size differs from its plan")
    return plan


def previous_media_inventory(plan: dict, root: Path) -> tuple[set[str], set[str]]:
    original_plan_path = root / "perception-av-event-media-plan-02" / "plan.json"
    if sha256_file(original_plan_path) != PLAN_SHA256:
        raise ValueError("original media plan hash changed")
    original_plan = json.loads(original_plan_path.read_text(encoding="utf-8"))
    hashes = set()
    video_ids = set()
    expected_manifests = {
        **MEDIA_MANIFESTS,
        "perception-av-event-train-media-01": TRAIN_MEDIA_MANIFEST_SHA256,
        "perception-av-event-valid-media-01": VALID_MEDIA_MANIFEST_SHA256,
    }
    for name, expected_sha in expected_manifests.items():
        directory = root / name
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError(f"pinned previous media manifest missing: {manifest_path}")
        if sha256_file(manifest_path) != expected_sha:
            raise ValueError(f"previous media manifest hash changed: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for row in manifest.get("clips", []):
            if row.get("sha256"):
                hashes.add(row["sha256"])
            video_ids.add(row["video_id"])
    for split in ("train", "valid"):
        video_ids.update(row["video_id"] for row in original_plan["splits"][split]["rows"])
    if original_plan.get("test_opened") is not False:
        raise ValueError("original plan is not test-closed")
    return hashes, video_ids


def acquire(plan_path: Path, expected_sha256: str, output: Path) -> dict:
    plan = validate_plan(plan_path, expected_sha256)
    if output.exists():
        raise ValueError("fresh output directory required")
    if shutil.disk_usage(output.parent).free < plan["total_uncompressed_bytes"] + RESERVE_BYTES:
        raise ValueError("insufficient disk space for media plus 8GB reserve")
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "RUNNING",
        "plan_sha256": expected_sha256,
        "plan_path": str(plan_path),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "split": "train_fresh_holdout",
        "test_opened": False,
        "model_loaded": False,
        "training": False,
        "gpu_used": False,
        "paid_compute_cost": 0,
        "clips": [],
        "started_unix": time.time(),
    }
    receipt = output / "manifest.json"
    write_receipt(receipt, report)
    existing_hashes, previous_ids = previous_media_inventory(plan, ROOT)
    if {row["video_id"] for row in plan["rows"]} & previous_ids:
        raise ValueError("fresh holdout IDs overlap an earlier selected/acquired media ID")
    try:
        pin = plan["archive_pin"]
        with zipfile.ZipFile(RemoteZip(pin["url"], pin["bytes"], pin["etag"])) as archive:
            for row in plan["rows"]:
                info = archive.getinfo(row["member"])
                if (info.file_size, info.compress_size, info.CRC) != (
                    row["bytes"], row["compressed_bytes"], row["crc32"]
                ):
                    raise ValueError(f"pinned archive member changed: {row['video_id']}")
                path = output / f"{row['video_id']}.mp4"
                digest = hashlib.sha256()
                size = 0
                with archive.open(info) as source, path.open("xb") as target:
                    while chunk := source.read(CHUNK_BYTES):
                        target.write(chunk)
                        digest.update(chunk)
                        size += len(chunk)
                file_hash = digest.hexdigest()
                if size != row["bytes"]:
                    raise ValueError(f"extracted length mismatch: {row['video_id']}")
                if file_hash in existing_hashes:
                    raise ValueError(
                        "exact media-byte duplicate with prior artifact: " f"{row['video_id']}"
                    )
                existing_hashes.add(file_hash)
                report["clips"].append(
                    {
                        "video_id": row["video_id"],
                        "path": str(path),
                        "bytes": size,
                        "sha256": file_hash,
                        "crc32": row["crc32"],
                        "member": row["member"],
                    }
                )
                write_receipt(receipt, report)
        if len(report["clips"]) != 200:
            raise ValueError("acquisition did not produce all planned clips")
        report["status"] = "PASS_ACQUIRED_UNVERIFIED_DECODE"
    except Exception as error:
        report["status"] = "INCOMPLETE"
        report["error"] = f"{type(error).__name__}: {error}"
        write_receipt(receipt, report)
        raise
    report["completed_unix"] = time.time()
    write_receipt(receipt, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = acquire(args.plan, args.plan_sha256, args.output)
    print(json.dumps({"status": result["status"], "clips": len(result["clips"])}, sort_keys=True))


if __name__ == "__main__":
    main()
