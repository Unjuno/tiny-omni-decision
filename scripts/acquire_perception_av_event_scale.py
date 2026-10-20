"""Acquire the pinned scene-disjoint scale cohort without loading a model."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
import zipfile
from pathlib import Path

import requests

from scripts.acquire_perception_av_event_fresh_holdout import previous_media_inventory
from scripts.audit_mmfit import RemoteZip
from scripts.audit_perception_av_correspondence import sha256_file
from scripts.plan_perception_av_event_media import (
    MAX_CLIP_BYTES,
    MAX_SPLIT_BYTES,
    TRAIN_ARCHIVE_PIN,
)

RESERVE_BYTES = 8_000_000_000
CHUNK_BYTES = 1024 * 1024
SPLIT_COUNTS = {"fit": 233, "selection": 58, "calibration": 39, "development_validation": 58}


class RetryingRemoteZip(RemoteZip):
    """Retry transient range-read TLS/connection failures without changing the range."""

    def __init__(self, url: str, size: int, etag: str):
        super().__init__(url, size, etag)
        self.retries = 0

    def read(self, size: int = -1) -> bytes:
        position = self.position
        for attempt in range(5):
            try:
                return super().read(size)
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as error:
                self.position = position
                if attempt == 4:
                    raise
                delay = min(2**attempt, 8)
                self.retries += 1
                print(
                    f"transient archive range error ({type(error).__name__}); "
                    f"retry {attempt + 1}/4 after {delay}s",
                    flush=True,
                )
                time.sleep(delay)
        raise RuntimeError("unreachable remote ZIP retry state")


def reconcile_existing_media(
    output: Path,
    rows: list[dict],
    recorded_clips: list[dict],
    existing_hashes: set[str],
) -> tuple[list[dict], list[dict]]:
    """Hash all files from an interrupted attempt and preserve partial files."""
    planned = {row["video_id"]: row for row in rows}
    recorded = {clip.get("video_id"): clip for clip in recorded_clips}
    if len(recorded) != len(recorded_clips) or not set(recorded) <= set(planned):
        raise ValueError("resume manifest contains duplicate or unplanned clip IDs")
    found: dict[str, dict] = {}
    partials = []
    for path in sorted(output.glob("*.mp4")):
        video_id = path.stem
        row = planned.get(video_id)
        if row is None:
            raise ValueError(f"resume directory has an unplanned media file: {path.name}")
        if path.stat().st_size != row["bytes"]:
            partial_path = path.with_name(path.name + ".partial-01")
            if partial_path.exists():
                raise ValueError(f"preserved partial path already exists: {partial_path.name}")
            old_size = path.stat().st_size
            path.replace(partial_path)
            partials.append({"name": partial_path.name, "bytes": old_size})
            continue
        file_hash = sha256_file(path)
        old_clip = recorded.get(video_id)
        if old_clip is not None and (
            old_clip.get("sha256") != file_hash
            or old_clip.get("bytes") != row["bytes"]
            or old_clip.get("model_split") != row["model_split"]
        ):
            raise ValueError(f"recorded resume receipt differs from media bytes: {video_id}")
        if file_hash in existing_hashes:
            raise ValueError(f"resume media duplicates an earlier source: {video_id}")
        existing_hashes.add(file_hash)
        found[video_id] = {
            "video_id": video_id,
            "model_split": row["model_split"],
            "path": str(path.resolve()),
            "bytes": row["bytes"],
            "sha256": file_hash,
            "crc32": row["crc32"],
            "member": row["member"],
        }
    if not set(recorded) <= set(found):
        raise ValueError("resume manifest references a missing or incomplete media file")
    clips = [found[row["video_id"]] for row in rows if row["video_id"] in found]
    return clips, partials


def validate_plan(plan_path: Path, expected_sha256: str) -> dict:
    if sha256_file(plan_path) != expected_sha256:
        raise ValueError("scale media plan hash mismatch")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    rows = plan.get("rows")
    if (
        plan.get("status") != "PASS_FRESH_VIDEO_SCALE_PLAN"
        or plan.get("training") is not False
        or plan.get("test_opened") is not False
        or plan.get("archive_pin") != TRAIN_ARCHIVE_PIN
        or not isinstance(rows, list)
        or len(rows) != sum(SPLIT_COUNTS.values())
    ):
        raise ValueError("plan is not the pinned TRAIN-only 388-video scale plan")
    ids = [row.get("video_id") for row in rows]
    if len(set(ids)) != len(ids) or any(not isinstance(video_id, str) for video_id in ids):
        raise ValueError("scale plan contains duplicate or invalid video IDs")
    counts = {split: 0 for split in SPLIT_COUNTS}
    event_counts = {split: 0 for split in SPLIT_COUNTS}
    total_bytes = 0
    for row in rows:
        split = row.get("model_split")
        if split not in counts:
            raise ValueError("scale plan row has an unknown model split")
        if (
            row.get("metadata", {}).get("split") != "train"
            or row.get("metadata", {}).get("video_id") != row.get("video_id")
            or row.get("member") != f"videos/{row.get('video_id')}.mp4"
            or type(row.get("candidate_events")) is not int
            or row["candidate_events"] < 1
            or type(row.get("bytes")) is not int
            or not 0 < row["bytes"] <= MAX_CLIP_BYTES
        ):
            raise ValueError("invalid TRAIN identity, source member, event count or clip size")
        counts[split] += 1
        event_counts[split] += row["candidate_events"]
        total_bytes += row["bytes"]
    if counts != SPLIT_COUNTS:
        raise ValueError("video-level split sizes differ from frozen plan")
    if total_bytes != plan.get("total_uncompressed_bytes") or total_bytes > MAX_SPLIT_BYTES:
        raise ValueError("scale media byte total differs from plan or exceeds limits")
    for split in SPLIT_COUNTS:
        summary = plan.get("split_summary", {}).get(split, {})
        if (
            summary.get("video_count") != counts[split]
            or summary.get("candidate_events") != event_counts[split]
        ):
            raise ValueError(f"{split} summary differs from frozen rows")
    if sum(event_counts.values()) != plan.get("candidate_events"):
        raise ValueError("total candidate event count differs from plan")
    return plan


def acquire(
    plan_path: Path, plan_sha256: str, output: Path, root: Path, resume: bool = False
) -> dict:
    plan = validate_plan(plan_path, plan_sha256)
    if output.exists() and not resume:
        raise ValueError("fresh output directory required")
    if resume and not output.is_dir():
        raise ValueError("resume requires an existing acquisition directory")

    original_hashes, original_ids = previous_media_inventory({}, root)
    holdout_dir = root / "perception-av-event-fresh-holdout-media-01"
    holdout_manifest_path = holdout_dir / "manifest.json"
    expected_holdout_manifest = "8591dbb0c173c70780c6d6066f18b2d7e3f29ba43c215ae0b2a251a07e2ac118"
    if sha256_file(holdout_manifest_path) != expected_holdout_manifest:
        raise ValueError("fixed-checkpoint holdout media manifest changed")
    holdout_manifest = json.loads(holdout_manifest_path.read_text(encoding="utf-8"))
    if (
        holdout_manifest.get("status") != "PASS_ACQUIRED_UNVERIFIED_DECODE"
        or len(holdout_manifest.get("clips", [])) != 200
    ):
        raise ValueError("fixed-checkpoint holdout media is incomplete")
    for clip in holdout_manifest["clips"]:
        original_ids.add(clip["video_id"])
        original_hashes.add(clip["sha256"])
    plan_ids = {row["video_id"] for row in plan["rows"]}
    if plan_ids & original_ids:
        raise ValueError("scale plan overlaps an earlier media ID")

    receipt_path = output / "manifest.json"
    if resume:
        old_report = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            old_report.get("plan_sha256") != plan_sha256
            or old_report.get("status") not in {"RUNNING", "INCOMPLETE"}
            or old_report.get("training") is not False
            or old_report.get("test_opened") is not False
        ):
            raise ValueError("resume receipt is not an incomplete attempt for this exact plan")
        recovered_clips, preserved_partials = reconcile_existing_media(
            output, plan["rows"], old_report.get("clips", []), original_hashes
        )
        report = old_report
        report.pop("error", None)
        report["clips"] = recovered_clips
        report["status"] = "RUNNING"
        report["resume_count"] = int(report.get("resume_count", 0)) + 1
        report["resumed_unix"] = time.time()
        report.setdefault("preserved_partial_files", []).extend(preserved_partials)
    else:
        output.mkdir(parents=True, exist_ok=False)
        recovered_clips = []
        report = {
            "status": "RUNNING",
            "plan_sha256": plan_sha256,
            "plan_path": str(plan_path.resolve()),
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "split": "train_scale",
            "training": False,
            "gpu_used": False,
            "test_opened": False,
            "paid_compute_cost": 0,
            "clips": [],
            "started_unix": time.time(),
            "resume_count": 0,
            "preserved_partial_files": [],
        }
    remaining_bytes = plan["total_uncompressed_bytes"] - sum(
        clip["bytes"] for clip in recovered_clips
    )
    if shutil.disk_usage(output.parent).free < remaining_bytes + RESERVE_BYTES:
        raise ValueError("insufficient disk space for remaining media plus 8GB reserve")

    def write_receipt() -> None:
        temporary = receipt_path.with_name(receipt_path.name + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(receipt_path)

    write_receipt()
    try:
        remote = RetryingRemoteZip(
            TRAIN_ARCHIVE_PIN["url"], TRAIN_ARCHIVE_PIN["bytes"], TRAIN_ARCHIVE_PIN["etag"]
        )
        with zipfile.ZipFile(remote) as archive:
            completed_ids = {clip["video_id"] for clip in report["clips"]}
            for row in plan["rows"]:
                if row["video_id"] in completed_ids:
                    continue
                info = archive.getinfo(row["member"])
                if (info.file_size, info.compress_size, info.CRC) != (
                    row["bytes"],
                    row["compressed_bytes"],
                    row["crc32"],
                ):
                    raise ValueError(f"pinned archive member changed: {row['video_id']}")
                media_path = output / f"{row['video_id']}.mp4"
                digest = hashlib.sha256()
                size = 0
                with archive.open(info) as source, media_path.open("xb") as target:
                    while chunk := source.read(CHUNK_BYTES):
                        target.write(chunk)
                        digest.update(chunk)
                        size += len(chunk)
                file_hash = digest.hexdigest()
                if size != row["bytes"]:
                    raise ValueError(f"extracted length mismatch: {row['video_id']}")
                if file_hash in original_hashes:
                    raise ValueError(
                        f"exact media-byte duplicate from earlier artifact: {row['video_id']}"
                    )
                original_hashes.add(file_hash)
                report["clips"].append(
                    {
                        "video_id": row["video_id"],
                        "model_split": row["model_split"],
                        "path": str(media_path.resolve()),
                        "bytes": size,
                        "sha256": file_hash,
                        "crc32": row["crc32"],
                        "member": row["member"],
                    }
                )
                if len(report["clips"]) % 10 == 0:
                    write_receipt()
                    print(f"acquired {len(report['clips'])}/388 videos", flush=True)
            report["network_range_retries"] = remote.retries
        if len(report["clips"]) != 388:
            raise ValueError("acquisition did not produce every planned clip")
        actual_split_counts = {
            split: sum(clip["model_split"] == split for clip in report["clips"])
            for split in SPLIT_COUNTS
        }
        if actual_split_counts != SPLIT_COUNTS:
            raise ValueError("acquired split grouping differs from plan")
        report["status"] = "PASS_ACQUIRED_UNVERIFIED_DECODE"
    except Exception as error:
        report["status"] = "INCOMPLETE"
        report["error"] = f"{type(error).__name__}: {error}"
        write_receipt()
        raise
    report["completed_unix"] = time.time()
    report["total_bytes"] = sum(clip["bytes"] for clip in report["clips"])
    write_receipt()
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("C:/CodexArtifacts/pretrained-reuse-r2"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = acquire(args.plan, args.plan_sha256, args.output, args.root, args.resume)
    print(
        json.dumps(
            {
                "status": result["status"],
                "videos": len(result["clips"]),
                "bytes": result["total_bytes"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
