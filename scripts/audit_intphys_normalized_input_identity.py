"""Hash IntPhys's exact sampled, normalized VideoPrism input using pinned TensorFlow."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

from scripts.audit_intphys_temporal_sampling_coverage import uniform_frame_numbers
from scripts.extract_frozen_videoprism_features import (
    FRAME_SIZE,
    _preprocess,
)
from scripts.extract_frozen_videoprism_features import (
    SOURCE_COMMIT as VIDEOPRISM_SOURCE_COMMIT,
)
from scripts.extract_intphys_train_features import (
    EXPECTED_FRAMES,
    canonical_status_hash,
    parse_train_member,
    rgb_sequence_sha256,
)
from scripts.intphys_train_archive_pins import TRAIN_ARCHIVE_PINS

SAMPLED_FRAME_NUMBERS = uniform_frame_numbers(EXPECTED_FRAMES, 8)
EXPECTED_FRAME_MASK = (1 << EXPECTED_FRAMES) - 1


def hash_normalized_frames(normalized_frames: Any, numpy_module: Any) -> str:
    """Hash exactly the contiguous float32 tensor bytes used by feature extraction."""
    return hashlib.sha256(numpy_module.ascontiguousarray(normalized_frames).tobytes()).hexdigest()


def merge_normalized_identities(
    base_records: list[dict[str, str]], normalized_by_sample_id: dict[str, str]
) -> list[dict[str, str]]:
    """Attach one valid normalized-input hash to each existing identity record."""
    base_ids = [row.get("sample_id", "") for row in base_records]
    if not base_ids or any(not sample_id for sample_id in base_ids):
        raise ValueError("base identity records must have nonempty sample IDs")
    if len(set(base_ids)) != len(base_ids):
        raise ValueError("base identity records contain duplicate sample IDs")
    if set(base_ids) != set(normalized_by_sample_id):
        raise ValueError("normalized identity sample IDs do not exactly match the base manifest")
    merged = []
    for row in base_records:
        digest = normalized_by_sample_id[row["sample_id"]]
        if len(digest) != 64:
            raise ValueError("normalized input identity must be a SHA-256 hex digest")
        try:
            bytes.fromhex(digest)
        except ValueError as error:
            raise ValueError("normalized input identity must be a SHA-256 hex digest") from error
        merged.append({**row, "normalized_input_sha256": digest})
    return merged


def _sha256_file(path: Path) -> tuple[int, str, str]:
    md5 = hashlib.md5(usedforsecurity=False)
    sha256 = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            size += len(block)
            md5.update(block)
            sha256.update(block)
    return size, md5.hexdigest(), sha256.hexdigest()


def _load_identity_manifest(path: Path, *, shard_number: int) -> list[dict[str, str]]:
    prefix = f"train.{shard_number}/"
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not str(row.get("sample_id", "")).startswith(prefix):
            raise ValueError(
                f"base identity manifest has an unexpected sample ID at line {line_number}"
            )
        rows.append({str(key): str(value) for key, value in row.items()})
    if not rows:
        raise ValueError("base identity manifest is empty")
    return rows


def audit_normalized_input_identity(
    archive_path: Path,
    base_identity_path: Path,
    *,
    shard_number: int,
    identity_output: Path,
    receipt_output: Path,
    expected_video_count: int = 3_750,
    tf_module: Any | None = None,
    numpy_module: Any | None = None,
) -> dict[str, Any]:
    """Recheck a pinned train shard and hash the exact source input tensor for each video."""
    if shard_number not in TRAIN_ARCHIVE_PINS:
        raise ValueError(f"unsupported IntPhys train shard: {shard_number}")
    if type(expected_video_count) is not int or expected_video_count < 1:
        raise ValueError("expected video count must be a positive integer")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    if tf_module is None:
        import tensorflow as tf_module  # type: ignore[no-redef]
    if numpy_module is None:
        import numpy as numpy_module  # type: ignore[no-redef]
    if platform.python_version() != "3.11.16":
        raise RuntimeError("normalized identity audit requires Python 3.11.16")
    if str(tf_module.__version__) != "2.18.1":
        raise RuntimeError("normalized identity audit requires TensorFlow 2.18.1")
    if str(numpy_module.__version__) != "1.26.4":
        raise RuntimeError("normalized identity audit requires NumPy 1.26.4")
    tensorflow_distribution_version = importlib.metadata.version("tensorflow_cpu")
    if tensorflow_distribution_version != "2.18.1":
        raise RuntimeError("normalized identity audit requires tensorflow-cpu 2.18.1")
    if tf_module.config.list_physical_devices("GPU"):
        raise RuntimeError("normalized identity audit must run with no visible GPU")

    archive_path = archive_path.resolve(strict=True)
    base_identity_path = base_identity_path.resolve(strict=True)
    base_records = _load_identity_manifest(base_identity_path, shard_number=shard_number)
    base_by_id = {row["sample_id"]: row for row in base_records}
    if len(base_by_id) != len(base_records):
        raise ValueError("base identity manifest has duplicate sample IDs")

    pin = TRAIN_ARCHIVE_PINS[shard_number]
    total_started = time.perf_counter()
    size, md5, archive_sha256 = _sha256_file(archive_path)
    if size != pin["bytes"] or md5 != pin["md5"]:
        raise ValueError("archive does not match the publisher's pinned size and MD5")

    rows: dict[str, dict[str, Any]] = {}
    seen_video_ids: set[str] = set()
    active_video_id: str | None = None
    member_count = 0
    status_count = 0

    def finish_video(video_id: str) -> None:
        row = rows[video_id]
        if row["status_hash"] is None or any(
            row["channel_masks"][channel] != EXPECTED_FRAME_MASK
            for channel in ("scene", "depth", "masks")
        ):
            raise ValueError(f"incomplete video structure for {video_id}")
        if any(number not in row["selected_frames"] for number in SAMPLED_FRAME_NUMBERS):
            raise ValueError(f"missing selected RGB frames for {video_id}")
        sample_id = f"train.{shard_number}/{video_id}"
        base = base_by_id.get(sample_id)
        if base is None:
            raise ValueError(f"video is absent from the base identity manifest: {sample_id}")
        rgb_hashes = row["rgb_frame_hashes"]
        if any(value is None for value in rgb_hashes):
            raise ValueError(f"incomplete RGB identity for {sample_id}")
        if row["status_hash"] != base.get("status_identity_sha256"):
            raise ValueError(f"status identity differs from base manifest: {sample_id}")
        if rgb_sequence_sha256(
            {index + 1: digest for index, digest in enumerate(rgb_hashes)}
        ) != base.get("rgb_sequence_sha256"):
            raise ValueError(f"full RGB identity differs from base manifest: {sample_id}")
        sampled = numpy_module.stack(
            [row["selected_frames"][number] for number in SAMPLED_FRAME_NUMBERS]
        )
        normalized = _preprocess(sampled, tf_module)
        row["normalized_input_sha256"] = hash_normalized_frames(normalized, numpy_module)

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
                    raise ValueError(f"video members are not contiguous: {video_id}")
                if active_video_id is not None:
                    finish_video(active_video_id)
                seen_video_ids.add(video_id)
                active_video_id = video_id
                rows[video_id] = {
                    "status_hash": None,
                    "channel_masks": {name: 0 for name in ("scene", "depth", "masks")},
                    "rgb_frame_hashes": [None] * EXPECTED_FRAMES,
                    "selected_frames": {},
                }
            row = rows[video_id]
            if channel == "status":
                payload = archive.extractfile(member)
                if payload is None or row["status_hash"] is not None:
                    raise ValueError(f"missing or duplicate status record: {video_id}")
                row["status_hash"] = canonical_status_hash(json.load(payload))
                status_count += 1
            elif channel in row["channel_masks"]:
                if frame_number is None or frame_number not in range(1, EXPECTED_FRAMES + 1):
                    raise ValueError(f"invalid frame number: {member.name}")
                bit = 1 << (frame_number - 1)
                if row["channel_masks"][channel] & bit:
                    raise ValueError(f"duplicate frame: {member.name}")
                row["channel_masks"][channel] |= bit
                if channel == "scene":
                    payload = archive.extractfile(member)
                    if payload is None:
                        raise ValueError(f"missing RGB payload: {member.name}")
                    frame_bytes = payload.read()
                    row["rgb_frame_hashes"][frame_number - 1] = hashlib.sha256(
                        frame_bytes
                    ).hexdigest()
                    if frame_number in SAMPLED_FRAME_NUMBERS:
                        decoded = tf_module.io.decode_png(frame_bytes, channels=3)
                        row["selected_frames"][frame_number] = numpy_module.asarray(decoded)
            else:
                raise ValueError(f"unexpected data member: {member.name}")
            if member_count % 100_000 == 0:
                print(
                    f"members={member_count} videos={len(rows)}",
                    file=sys.stderr,
                    flush=True,
                )
        if active_video_id is not None:
            finish_video(active_video_id)

    if len(rows) != expected_video_count or status_count != expected_video_count:
        raise ValueError(
            f"expected {expected_video_count} videos/statuses, observed {len(rows)}/{status_count}"
        )
    normalized_by_id = {
        f"train.{shard_number}/{video_id}": row["normalized_input_sha256"]
        for video_id, row in rows.items()
    }
    merged_records = merge_normalized_identities(base_records, normalized_by_id)

    identity_output = identity_output.resolve()
    receipt_output = receipt_output.resolve()
    if identity_output == receipt_output:
        raise ValueError("identity and receipt output paths must differ")
    identity_output.parent.mkdir(parents=True, exist_ok=True)
    receipt_output.parent.mkdir(parents=True, exist_ok=True)
    with identity_output.open("x", encoding="utf-8", newline="\n") as destination:
        for identity in merged_records:
            destination.write(json.dumps(identity, sort_keys=True) + "\n")
    script_path = Path(__file__).resolve()
    preprocess_path = Path(__file__).with_name("extract_frozen_videoprism_features.py")
    receipt = {
        "status": "pass_exact_normalized_input_identity_audit",
        "shard_number": shard_number,
        "archive_path": str(archive_path),
        "archive_bytes": size,
        "archive_md5": md5,
        "archive_sha256": archive_sha256,
        "publisher_pin_verified": True,
        "videos": len(rows),
        "status_records": status_count,
        "tar_members": member_count,
        "sampled_frame_numbers_1_based": SAMPLED_FRAME_NUMBERS,
        "video_prism_input_frame_size": FRAME_SIZE,
        "sample_id_order_sha256": hashlib.sha256(
            "\n".join(sorted(normalized_by_id)).encode()
        ).hexdigest(),
        "base_identity_manifest_path": str(base_identity_path),
        "base_identity_manifest_sha256": hashlib.sha256(
            base_identity_path.read_bytes()
        ).hexdigest(),
        "identity_manifest_path": str(identity_output),
        "identity_manifest_sha256": hashlib.sha256(identity_output.read_bytes()).hexdigest(),
        "python_version": platform.python_version(),
        "tensorflow_version": str(tf_module.__version__),
        "numpy_version": str(numpy_module.__version__),
        "tensorflow_distribution_version": tensorflow_distribution_version,
        "tensorflow_visible_gpu_devices": 0,
        "videoprism_preprocess_source_commit": VIDEOPRISM_SOURCE_COMMIT,
        "videoprism_preprocess_source_sha256": hashlib.sha256(
            preprocess_path.read_bytes()
        ).hexdigest(),
        "audit_script_sha256": hashlib.sha256(script_path.read_bytes()).hexdigest(),
        "normalized_input_hash": (
            "SHA-256 of contiguous float32 bytes returned by the exact _preprocess "
            "function used during frozen feature extraction"
        ),
        "elapsed_seconds": round(time.perf_counter() - total_started, 3),
        "validation_or_sealed_data_accessed": False,
        "model_or_gpu_used": False,
    }
    with receipt_output.open("x", encoding="utf-8", newline="\n") as destination:
        destination.write(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    receipt["receipt_sha256"] = hashlib.sha256(receipt_output.read_bytes()).hexdigest()
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--base-identity-manifest", type=Path, required=True)
    parser.add_argument("--shard-number", type=int, choices=TRAIN_ARCHIVE_PINS, required=True)
    parser.add_argument("--identity-output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_normalized_input_identity(
        args.archive,
        args.base_identity_manifest,
        shard_number=args.shard_number,
        identity_output=args.identity_output,
        receipt_output=args.receipt_output,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
