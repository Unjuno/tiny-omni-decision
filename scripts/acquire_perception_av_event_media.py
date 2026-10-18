"""Acquire pinned train/development media for AV event-source selection."""

import argparse
import json
import shutil
import subprocess
import time
import zipfile
from pathlib import Path

from scripts.audit_mmfit import RemoteZip
from scripts.audit_perception_av_correspondence import sha256_file
from scripts.plan_perception_av_event_media import (
    MAX_CLIP_BYTES,
    MAX_SPLIT_BYTES,
    MEDIA_MANIFESTS,
    ROOT,
    TRAIN_ARCHIVE_PIN,
    VALID_ARCHIVE_PIN,
)

RESERVE_BYTES = 8_000_000_000


def write_receipt(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def validate_plan(plan, split):
    if plan.get("status") != "PASS_FRESH_EVENT_MEDIA_PLAN" or plan.get("test_opened") is not False:
        raise ValueError("source media plan must be a passing train/valid-only plan")
    if plan.get("training") is not False or plan.get("gpu_used") is not False:
        raise ValueError("plan must not imply model training or GPU use")
    if plan.get("previous_media_manifest_sha256") != MEDIA_MANIFESTS:
        raise ValueError("previous-media inventory pins changed")
    entry = plan.get("splits", {}).get(split)
    expected_pin = TRAIN_ARCHIVE_PIN if split == "train" else VALID_ARCHIVE_PIN
    expected_media_count = 400
    if not isinstance(entry, dict) or entry.get("pin") != expected_pin:
        raise ValueError(f"unexpected {split} archive pin")
    rows = entry.get("rows")
    if not isinstance(rows, list) or len(rows) != expected_media_count:
        raise ValueError(f"expected {expected_media_count} {split} media records")
    if len({row.get("video_id") for row in rows}) != len(rows):
        raise ValueError("duplicate media video ID")
    if any(
        row.get("metadata", {}).get("split") != split
        or row.get("metadata", {}).get("video_id") != row.get("video_id")
        or not str(row.get("video_id", "")).startswith("video_")
        or not 0 < row.get("bytes", 0) <= MAX_CLIP_BYTES
        for row in rows
    ):
        raise ValueError("invalid split, identity or media byte limit")
    split_bytes = sum(row["bytes"] for row in rows)
    if split_bytes != entry.get("total_uncompressed_bytes") or split_bytes > MAX_SPLIT_BYTES:
        raise ValueError("split media size does not match the plan or exceeds cap")
    if split == "train":
        model_split_counts = {
            name: sum(row.get("model_split") == name for row in rows)
            for name in ("train", "calibration")
        }
        if model_split_counts != {"train": 300, "calibration": 100}:
            raise ValueError("train/calibration groups must be 300/100 videos")
    elif any(row.get("model_split") != "development_validation" for row in rows):
        raise ValueError("official valid videos must remain development-only")
    return entry


def acquire(output, plan_path, expected_plan_sha256, split):
    if output.exists():
        raise ValueError("fresh output directory required")
    if sha256_file(plan_path) != expected_plan_sha256:
        raise ValueError("plan hash mismatch")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    entry = validate_plan(plan, split)
    remaining_total = sum(
        split_entry["total_uncompressed_bytes"] for split_entry in plan["splits"].values()
    )
    if shutil.disk_usage(ROOT).free < remaining_total + RESERVE_BYTES:
        raise ValueError("insufficient disk space for both splits plus 8GB reserve")

    output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "RUNNING",
        "task_id": plan["task_id"],
        "split": split,
        "plan_sha256": expected_plan_sha256,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "runner_sha256": sha256_file(Path(__file__)),
        "archive_pin": entry["pin"],
        "clips": [],
        "test_opened": False,
        "model_loaded": False,
        "training": False,
        "gpu_used": False,
        "paid_compute_cost": 0,
        "started_unix": time.time(),
    }
    manifest_path = output / "manifest.json"
    write_receipt(manifest_path, report)
    started = time.monotonic()
    try:
        selected = entry["rows"]
        with zipfile.ZipFile(
            RemoteZip(entry["pin"]["url"], entry["pin"]["bytes"], entry["pin"]["etag"])
        ) as archive:
            for index, row in enumerate(selected):
                video_id = row["video_id"]
                if Path(row["member"]).name != video_id + ".mp4":
                    raise ValueError(f"unexpected archive member path for {video_id}")
                info = archive.getinfo(row["member"])
                if (info.file_size, info.compress_size, info.CRC) != (
                    row["bytes"],
                    row["compressed_bytes"],
                    row["crc32"],
                ):
                    raise ValueError(f"archive member metadata changed for {video_id}")
                remaining = sum(item["bytes"] for item in selected[index:])
                if shutil.disk_usage(ROOT).free < remaining + RESERVE_BYTES:
                    raise ValueError("8GB disk reserve would be violated")
                media_path = output / (video_id + ".mp4")
                partial_path = output / (video_id + ".mp4.partial")
                if media_path.exists() or partial_path.exists():
                    raise ValueError(f"unexpected existing media path for {video_id}")
                digest_path = partial_path
                with archive.open(info) as reader, digest_path.open("xb") as writer:
                    shutil.copyfileobj(reader, writer, length=1024 * 1024)
                if digest_path.stat().st_size != row["bytes"]:
                    raise ValueError(f"download length mismatch for {video_id}")
                media_sha256 = sha256_file(digest_path)
                digest_path.replace(media_path)
                report["clips"].append(
                    {
                        **row,
                        "path": str(media_path),
                        "sha256": media_sha256,
                        "reused": False,
                    }
                )
                report["elapsed_seconds"] = time.monotonic() - started
                write_receipt(manifest_path, report)
                print(
                    json.dumps(
                        {"split": split, "complete": len(report["clips"]), "total": len(selected)}
                    ),
                    flush=True,
                )
        report["status"] = "PASS_ACQUISITION_NOT_DECODE"
    except BaseException as error:
        report.update(status="INCOMPLETE", error=repr(error))
        raise
    finally:
        report["ended_unix"] = time.time()
        report["elapsed_seconds"] = time.monotonic() - started
        write_receipt(manifest_path, report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--split", choices=("train", "valid"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    acquire(args.output, args.plan, args.plan_sha256, args.split)


if __name__ == "__main__":
    main()
