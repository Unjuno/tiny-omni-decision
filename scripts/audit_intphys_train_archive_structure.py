"""Verify a pinned IntPhys train shard and its complete per-video structure."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
import time
from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

from scripts.audit_intphys_temporal_sampling_coverage import uniform_frame_numbers
from scripts.extract_intphys_train_features import (
    EXPECTED_FRAMES,
    canonical_status_hash,
    parse_train_member,
)
from scripts.intphys_train_archive_pins import TRAIN_ARCHIVE_PINS

_EXPECTED_FRAME_MASK = (1 << EXPECTED_FRAMES) - 1
_SAMPLED_FRAME_NUMBERS = uniform_frame_numbers(EXPECTED_FRAMES, 8)
_SAMPLED_FRAME_INDEX = {number: index for index, number in enumerate(_SAMPLED_FRAME_NUMBERS)}


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
                {
                    "status_hash": None,
                    "scene": 0,
                    "depth": 0,
                    "masks": 0,
                    "rgb_frame_hashes": [None] * EXPECTED_FRAMES,
                    "sampled_rgb_payload_hashes": [None] * len(_SAMPLED_FRAME_NUMBERS),
                    "sampled_rgb_pixel_hashes": [None] * len(_SAMPLED_FRAME_NUMBERS),
                },
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
                if channel == "scene":
                    payload = archive.extractfile(member)
                    if payload is None:
                        raise ValueError(f"missing RGB frame payload: {member.name}")
                    row["rgb_frame_hashes"][frame_number - 1] = hashlib.sha256(
                        frame_bytes := payload.read()
                    ).hexdigest()
                    if frame_number in _SAMPLED_FRAME_INDEX:
                        sampled_index = _SAMPLED_FRAME_INDEX[frame_number]
                        row["sampled_rgb_payload_hashes"][sampled_index] = hashlib.sha256(
                            frame_bytes
                        ).hexdigest()
                        with Image.open(BytesIO(frame_bytes)) as image:
                            decoded = image.convert("RGB")
                            width, height = decoded.size
                            pixel_digest = hashlib.sha256()
                            pixel_digest.update(width.to_bytes(4, "little"))
                            pixel_digest.update(height.to_bytes(4, "little"))
                            pixel_digest.update(decoded.tobytes())
                            row["sampled_rgb_pixel_hashes"][sampled_index] = (
                                pixel_digest.hexdigest()
                            )
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
    identity_records = []
    for sample_id in sample_ids:
        video_id = sample_id.split("/", 1)[1]
        row = rows[video_id]
        frame_hashes = row["rgb_frame_hashes"]
        if any(value is None for value in frame_hashes):
            raise ValueError(f"RGB identity is incomplete for {sample_id}")
        sampled_payload_hashes = row["sampled_rgb_payload_hashes"]
        sampled_pixel_hashes = row["sampled_rgb_pixel_hashes"]
        if any(value is None for value in (*sampled_payload_hashes, *sampled_pixel_hashes)):
            raise ValueError(f"sampled RGB identity is incomplete for {sample_id}")
        rgb_digest = hashlib.sha256()
        for value in frame_hashes:
            rgb_digest.update(bytes.fromhex(value))
        row["rgb_sequence_sha256"] = rgb_digest.hexdigest()
        sampled_payload_digest = hashlib.sha256()
        sampled_pixel_digest = hashlib.sha256()
        for value in sampled_payload_hashes:
            sampled_payload_digest.update(bytes.fromhex(value))
        for value in sampled_pixel_hashes:
            sampled_pixel_digest.update(bytes.fromhex(value))
        identity_records.append(
            {
                "sample_id": sample_id,
                "status_identity_sha256": row["status_hash"],
                "rgb_sequence_sha256": row["rgb_sequence_sha256"],
                "sampled_rgb_payload_sha256": sampled_payload_digest.hexdigest(),
                "sampled_decoded_rgb_pixels_sha256": sampled_pixel_digest.hexdigest(),
            }
        )
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
        "unique_rgb_sequence_hashes": len(
            {row["rgb_sequence_sha256"] for row in identity_records}
        ),
        "unique_sampled_rgb_payload_hashes": len(
            {row["sampled_rgb_payload_sha256"] for row in identity_records}
        ),
        "unique_sampled_decoded_rgb_pixel_hashes": len(
            {row["sampled_decoded_rgb_pixels_sha256"] for row in identity_records}
        ),
        "scene_frames": sum(row["scene"].bit_count() for row in rows.values()),
        "depth_frames": sum(row["depth"].bit_count() for row in rows.values()),
        "mask_frames": sum(row["masks"].bit_count() for row in rows.values()),
        "sample_id_order_sha256": hashlib.sha256("\n".join(sample_ids).encode()).hexdigest(),
        "validation_or_sealed_data_accessed": False,
        "model_or_gpu_used": False,
        "structure_scan_seconds": round(time.perf_counter() - started, 3),
        "total_elapsed_seconds": round(time.perf_counter() - total_started, 3),
        "identity_records": identity_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--shard-number", type=int, choices=TRAIN_ARCHIVE_PINS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--identity-output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_train_archive(
        args.archive,
        shard_number=args.shard_number,
        progress=lambda members, videos: print(
            f"members={members} videos={videos}", file=sys.stderr, flush=True
        ),
    )
    identities = result.pop("identity_records")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.identity_output.parent.mkdir(parents=True, exist_ok=True)
    with args.identity_output.open("x", encoding="utf-8", newline="\n") as destination:
        for identity in identities:
            destination.write(json.dumps(identity, sort_keys=True) + "\n")
    result["identity_manifest_path"] = str(args.identity_output.resolve())
    result["identity_manifest_sha256"] = hashlib.sha256(
        args.identity_output.read_bytes()
    ).hexdigest()
    with args.output.open("x", encoding="utf-8", newline="\n") as destination:
        destination.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    result["receipt_sha256"] = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
