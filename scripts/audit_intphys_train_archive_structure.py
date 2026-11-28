"""Verify a pinned IntPhys train shard and its complete per-video structure."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from scripts.extract_intphys_train_features import (
    EXPECTED_FRAMES,
    canonical_status_hash,
    parse_train_member,
)
from scripts.intphys_train_archive_pins import TRAIN_ARCHIVE_PINS

_EXPECTED_FRAME_MASK = (1 << EXPECTED_FRAMES) - 1


def _digest_file(path: Path) -> tuple[int, str, str]:
    md5 = hashlib.md5(usedforsecurity=False)
    sha256 = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            size += len(block)
            md5.update(block)
            sha256.update(block)
    return size, md5.hexdigest(), sha256.hexdigest()


def audit_train_archive(
    archive_path: Path,
    *,
    shard_number: int,
    expected_video_count: int = 3_750,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Verify publisher digest and each train video has one status and 100 frames/channel."""
    if shard_number not in TRAIN_ARCHIVE_PINS:
        raise ValueError(f"unsupported IntPhys train shard: {shard_number}")
    if type(expected_video_count) is not int or expected_video_count < 1:
        raise ValueError("expected video count must be a positive integer")
    archive_path = archive_path.resolve(strict=True)
    pin = TRAIN_ARCHIVE_PINS[shard_number]
    total_started = time.perf_counter()
    hash_started = time.perf_counter()
    size, md5, sha256 = _digest_file(archive_path)
    hashing_seconds = time.perf_counter() - hash_started
    if size != pin["bytes"]:
        raise ValueError(f"archive size mismatch: expected {pin['bytes']}, got {size}")
    if md5 != pin["md5"]:
        raise ValueError(f"archive MD5 mismatch: expected {pin['md5']}, got {md5}")

    started = time.perf_counter()
    rows: dict[str, dict[str, Any]] = {}
    seen_video_ids: set[str] = set()
    active_video_id: str | None = None
    member_count = 0
    channels = ("scene", "depth", "masks")
    with tarfile.open(archive_path, mode="r|gz") as archive:
        for member in archive:
            member_count += 1
            if not member.isfile():
                if not member.isdir():
                    raise ValueError(f"unsupported tar member: {member.name}")
                parse_train_member(member.name.rstrip("/"))
                continue

            video_id, channel, frame_number = parse_train_member(member.name)
            if video_id != active_video_id:
                if video_id in seen_video_ids:
                    raise ValueError(f"video members are not contiguous in archive: {video_id}")
                seen_video_ids.add(video_id)
                active_video_id = video_id
            row = rows.setdefault(
                video_id,
                {"status_hash": None, "scene": 0, "depth": 0, "masks": 0},
            )
            if channel == "status":
                if row["status_hash"] is not None:
                    raise ValueError(f"duplicate status record: {video_id}")
                payload = archive.extractfile(member)
                if payload is None:
                    raise ValueError(f"missing status payload: {video_id}")
                status = json.load(payload)
                row["status_hash"] = canonical_status_hash(status)
            elif channel in channels:
                if frame_number is None or frame_number not in range(1, EXPECTED_FRAMES + 1):
                    raise ValueError(f"invalid frame index: {member.name}")
                frame_bit = 1 << (frame_number - 1)
                if row[channel] & frame_bit:
                    raise ValueError(f"duplicate frame: {member.name}")
                row[channel] |= frame_bit
            else:
                raise ValueError(f"unexpected data member: {member.name}")
            if progress is not None and member_count % 100_000 == 0:
                progress(member_count, len(rows))

    incomplete = [
        video_id
        for video_id, row in rows.items()
        if row["status_hash"] is None
        or any(row[channel] != _EXPECTED_FRAME_MASK for channel in channels)
    ]
    if incomplete:
        raise ValueError(f"{len(incomplete)} videos have missing/incomplete data: {incomplete[:5]}")
    if len(rows) != expected_video_count:
        raise ValueError(
            f"expected {expected_video_count} train videos, observed {len(rows)}"
        )

    sample_ids = sorted(f"train.{shard_number}/{video_id}" for video_id in rows)
    return {
        "status": "pass_train_shard_structure_audit",
        "shard_number": shard_number,
        "archive_path": str(archive_path),
        "archive_bytes": size,
        "archive_md5": md5,
        "archive_sha256": sha256,
        "publisher_pin_verified": True,
        "archive_hashing_seconds": round(hashing_seconds, 3),
        "tar_members": member_count,
        "videos": len(rows),
        "status_records": len(rows),
        "all_train_labels_possible": True,
        "unique_status_hashes": len({row["status_hash"] for row in rows.values()}),
        "scene_frames": sum(row["scene"].bit_count() for row in rows.values()),
        "depth_frames": sum(row["depth"].bit_count() for row in rows.values()),
        "mask_frames": sum(row["masks"].bit_count() for row in rows.values()),
        "sample_id_order_sha256": hashlib.sha256("\n".join(sample_ids).encode()).hexdigest(),
        "validation_or_sealed_data_accessed": False,
        "model_or_gpu_used": False,
        "structure_scan_seconds": round(time.perf_counter() - started, 3),
        "total_elapsed_seconds": round(time.perf_counter() - total_started, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--shard-number", type=int, choices=TRAIN_ARCHIVE_PINS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_train_archive(
        args.archive,
        shard_number=args.shard_number,
        progress=lambda members, videos: print(
            f"members={members} videos={videos}", file=sys.stderr, flush=True
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as destination:
        destination.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    result["receipt_sha256"] = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
