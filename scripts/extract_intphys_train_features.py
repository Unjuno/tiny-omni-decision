"""Audit and stream VideoPrism features from one official IntPhys train shard."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import tarfile
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from scripts.audit_intphys_temporal_sampling_coverage import uniform_frame_numbers
from scripts.audit_videoprism_checkpoint import CHECKPOINT_SHA256, MODEL_REVISION
from scripts.extract_frozen_videoprism_features import (
    FRAME_COUNT,
    FRAME_SIZE,
    _preprocess,
)
from scripts.extract_frozen_videoprism_features import (
    SOURCE_COMMIT as VIDEOPRISM_SOURCE_COMMIT,
)

EXPECTED_ARCHIVE_BYTES = 37_016_506_106
EXPECTED_ARCHIVE_MD5 = "3428bcf398441a489111528634530212"
EXPECTED_FRAMES = 100
_FRAME = re.compile(r"^(scene|depth|masks)_(\d{3})\.png$")


def parse_train_member(name: str) -> tuple[str, str | None, int | None]:
    """Parse the publisher's `train/<video-id>/...` path without extracting it."""
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != "train":
        raise ValueError(f"unsafe or unexpected train archive path: {name}")
    if len(path.parts) == 1:
        return "", None, None
    video_id = path.parts[1]
    if len(path.parts) == 2:
        return video_id, None, None
    if len(path.parts) == 3 and path.name == "status.json":
        return video_id, "status", None
    if len(path.parts) == 4 and path.parts[2] in {"scene", "depth", "masks"}:
        match = _FRAME.fullmatch(path.name)
        if match is None or match.group(1) != path.parts[2]:
            raise ValueError(f"unexpected frame member: {name}")
        return video_id, match.group(1), int(match.group(2))
    if len(path.parts) == 3 and path.parts[2] in {"scene", "depth", "masks"}:
        return video_id, None, None
    raise ValueError(f"unexpected train archive member: {name}")


def canonical_status_hash(status: dict[str, Any]) -> str:
    """Hash normalized complete scene metadata so identical scenes group together."""
    if status.get("header", {}).get("is_possible") is not True:
        raise ValueError("IntPhys training status must mark every sample possible")
    if not isinstance(status.get("frames"), list) or len(status["frames"]) != EXPECTED_FRAMES:
        raise ValueError("IntPhys status must describe exactly 100 frames")
    raw = json.dumps(status, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def rgb_sequence_sha256(frame_hashes: dict[int, str]) -> str:
    """Hash all 100 original RGB frame payloads in temporal order."""
    expected = set(range(1, EXPECTED_FRAMES + 1))
    if set(frame_hashes) != expected:
        raise ValueError("RGB sequence identity requires exactly frames 1..100")
    digest = hashlib.sha256()
    for frame_number in range(1, EXPECTED_FRAMES + 1):
        try:
            frame_digest = bytes.fromhex(frame_hashes[frame_number])
        except (ValueError, TypeError) as error:
            raise ValueError("RGB frame identity must be a SHA-256 hex digest") from error
        if len(frame_digest) != hashlib.sha256().digest_size:
            raise ValueError("RGB frame identity must be a SHA-256 hex digest")
        digest.update(frame_digest)
    return digest.hexdigest()


def _reject_nonfinite_json_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON number: {value}")


def deterministic_group_split(
    sample_ids: list[str],
    status_hashes: list[str],
    media_hashes: list[str],
    *,
    sampled_media_hashes: list[str] | None = None,
    seed: int = 17,
    validation_buckets: int = 5,
) -> tuple[list[int], list[int], list[str]]:
    """Group duplicate metadata, full RGB sequences, or sampled model inputs."""
    if not sample_ids or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("sample IDs must be nonempty and unique")
    if len(status_hashes) != len(sample_ids) or len(media_hashes) != len(sample_ids):
        raise ValueError("identity hashes must match sample ID count")
    if sampled_media_hashes is not None and len(sampled_media_hashes) != len(sample_ids):
        raise ValueError("sampled-media identity hashes must match sample ID count")
    if validation_buckets < 2:
        raise ValueError("validation_buckets must be at least two")

    parent = list(range(len(sample_ids)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    first_seen: dict[tuple[str, str], int] = {}
    identities_by_video = zip(status_hashes, media_hashes, strict=True)
    for index, (status_hash, full_rgb_hash) in enumerate(identities_by_video):
        identities = [("status", status_hash), ("full-rgb", full_rgb_hash)]
        if sampled_media_hashes is not None:
            identities.append(("sampled-rgb", sampled_media_hashes[index]))
        for identity_kind, identity in identities:
            key = (identity_kind, identity)
            previous = first_seen.setdefault(key, index)
            union(index, previous)

    members_by_root: dict[int, list[int]] = defaultdict(list)
    for index in range(len(sample_ids)):
        members_by_root[find(index)].append(index)

    group_ids = [""] * len(sample_ids)
    train_indices: list[int] = []
    validation_indices: list[int] = []
    for indices in members_by_root.values():
        group_id = min(sample_ids[index] for index in indices)
        for index in indices:
            group_ids[index] = group_id
        split_key = f"intphys-train-group-v1:{seed}:{group_id}".encode()
        bucket = int.from_bytes(hashlib.sha256(split_key).digest()[:8], "big") % validation_buckets
        destination = validation_indices if bucket == 0 else train_indices
        destination.extend(indices)

    if not train_indices or not validation_indices:
        raise ValueError("deterministic grouping did not produce nonempty train and validation")
    train_groups = {group_ids[index] for index in train_indices}
    validation_groups = {group_ids[index] for index in validation_indices}
    if train_groups & validation_groups:
        raise ValueError("underlying scene/media groups overlap across internal splits")
    return sorted(train_indices), sorted(validation_indices), group_ids


def load_reference_split_assignments(
    train_features_path: Path, validation_features_path: Path
) -> tuple[dict[str, tuple[str, str]], dict[str, list[str]], dict[str, str]]:
    """Load immutable train/validation membership from a prior feature split."""
    assignments: dict[str, tuple[str, str]] = {}
    ordered_ids: dict[str, list[str]] = {}
    path_hashes: dict[str, str] = {}
    for split_name, path in (
        ("train", train_features_path),
        ("validation", validation_features_path),
    ):
        with np.load(path, allow_pickle=False) as payload:
            required = {"sample_ids", "group_ids"}
            if not required.issubset(payload.files):
                raise ValueError(f"reference {split_name} cache lacks sample/group identities")
            sample_ids = payload["sample_ids"].astype(str).tolist()
            group_ids = payload["group_ids"].astype(str).tolist()
        if (
            not sample_ids
            or len(sample_ids) != len(group_ids)
            or len(set(sample_ids)) != len(sample_ids)
            or any(not value for value in (*sample_ids, *group_ids))
        ):
            raise ValueError(f"reference {split_name} cache has invalid identities")
        ordered_ids[split_name] = sample_ids
        path_hashes[split_name] = sha256_file(path)
        for sample_id, group_id in zip(sample_ids, group_ids, strict=True):
            if sample_id in assignments:
                raise ValueError(f"reference split repeats sample ID: {sample_id}")
            assignments[sample_id] = (split_name, group_id)
    train_groups = {assignments[sample_id][1] for sample_id in ordered_ids["train"]}
    validation_groups = {
        assignments[sample_id][1] for sample_id in ordered_ids["validation"]
    }
    if train_groups & validation_groups:
        raise ValueError("reference split has cross-split group overlap")
    return assignments, ordered_ids, path_hashes


def apply_reference_split(
    sample_ids: list[str],
    reference_train_path: Path,
    reference_validation_path: Path,
) -> tuple[list[int], list[int], list[str], dict[str, str]]:
    """Reuse exact sample IDs, source order, and group assignment from a frozen split."""
    assignments, ordered_ids, path_hashes = load_reference_split_assignments(
        reference_train_path, reference_validation_path
    )
    if len(sample_ids) != len(set(sample_ids)) or set(sample_ids) != set(assignments):
        missing = sorted(set(assignments) - set(sample_ids))[:5]
        unexpected = sorted(set(sample_ids) - set(assignments))[:5]
        raise ValueError(
            f"current archive IDs differ from reference split; missing={missing}, "
            f"unexpected={unexpected}"
        )
    train_indices = [
        index for index, sample_id in enumerate(sample_ids)
        if assignments[sample_id][0] == "train"
    ]
    validation_indices = [
        index for index, sample_id in enumerate(sample_ids)
        if assignments[sample_id][0] == "validation"
    ]
    actual_order = {
        "train": [sample_ids[index] for index in train_indices],
        "validation": [sample_ids[index] for index in validation_indices],
    }
    if actual_order != ordered_ids:
        raise ValueError("archive sample order differs from frozen reference split ordering")
    group_ids = [assignments[sample_id][1] for sample_id in sample_ids]
    return train_indices, validation_indices, group_ids, path_hashes


def file_digests(path: Path) -> tuple[str, str, int]:
    md5 = hashlib.md5()
    sha256 = hashlib.sha256()
    byte_count = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            byte_count += len(chunk)
            md5.update(chunk)
            sha256.update(chunk)
    return md5.hexdigest(), sha256.hexdigest(), byte_count


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def module_version(module: Any, distribution_name: str) -> str:
    """Prefer the imported runtime version; Windows wheels may use another dist name."""
    version = getattr(module, "__version__", None)
    if isinstance(version, str) and version:
        return version
    return importlib.metadata.version(distribution_name)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--frame-count", type=int, default=FRAME_COUNT)
    parser.add_argument("--reference-train-features", type=Path)
    parser.add_argument("--reference-validation-features", type=Path)
    return parser.parse_args()


def main() -> None:
    run_started_at = datetime.now(UTC).isoformat()
    run_started = time.perf_counter()
    args = _args()
    if not 2 <= args.frame_count <= EXPECTED_FRAMES:
        raise ValueError(f"frame count must be between 2 and {EXPECTED_FRAMES}")
    if bool(args.reference_train_features) != bool(args.reference_validation_features):
        raise ValueError("both reference train and validation feature paths are required together")
    selected_frame_numbers = uniform_frame_numbers(EXPECTED_FRAMES, args.frame_count)
    archive_path = args.archive.resolve(strict=True)
    checkpoint = args.checkpoint.resolve(strict=True)
    source_root = args.source_root.resolve(strict=True)
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing feature directory: {output_dir}")
    archive_hash_started = time.perf_counter()
    if archive_path.stat().st_size != EXPECTED_ARCHIVE_BYTES:
        raise ValueError("train shard byte size differs from the official published size")
    archive_md5, archive_sha256, archive_bytes = file_digests(archive_path)
    archive_hash_seconds = time.perf_counter() - archive_hash_started
    if archive_md5 != EXPECTED_ARCHIVE_MD5:
        raise ValueError(f"train shard MD5 mismatch: {archive_md5}")
    if archive_bytes != EXPECTED_ARCHIVE_BYTES:
        raise ValueError("train shard changed while hashes were being computed")
    checkpoint_sha256 = sha256_file(checkpoint)
    if checkpoint_sha256 != CHECKPOINT_SHA256:
        raise ValueError("VideoPrism checkpoint SHA-256 differs from the pinned R1 artifact")
    source_commit = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if source_commit != VIDEOPRISM_SOURCE_COMMIT:
        raise ValueError(f"unexpected VideoPrism source commit: {source_commit}")

    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    sys.path.insert(0, str(source_root))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    import jax
    import jax.numpy as jnp
    import psutil
    import tensorflow as tf
    from videoprism import models as vp

    from tiny_omni_decision.temporal_tokens import pool_spatial_tokens_per_frame

    devices = jax.devices()
    if not devices or any(device.platform != "cpu" for device in devices):
        raise RuntimeError(f"CPU-only feature extraction required, found devices: {devices}")
    process = psutil.Process()
    host_memory_before = psutil.virtual_memory()
    process_memory_before = process.memory_info()
    model_load_started = time.perf_counter()
    model = vp.get_model("videoprism_public_v1_base")
    variables = vp.load_pretrained_weights(
        "videoprism_public_v1_base", checkpoint_path=str(checkpoint)
    )
    encode = jax.jit(lambda video: model.apply(variables, video, train=False))
    model_load_seconds = time.perf_counter() - model_load_started

    extraction_started = time.perf_counter()
    feature_rows: list[np.ndarray] = []
    sample_ids: list[str] = []
    status_hashes: list[str] = []
    media_hashes: list[str] = []
    sampled_media_hashes: list[str] = []
    forward_seconds: list[float] = []
    channel_counts: dict[str, int] = defaultdict(int)
    seen_video_ids: set[str] = set()
    active_video_id: str | None = None
    active: dict[str, Any] | None = None

    def finish_video(video_id: str, row: dict[str, Any]) -> None:
        expected_numbers = set(range(1, EXPECTED_FRAMES + 1))
        for channel in ("scene", "depth", "masks"):
            if row["frames"][channel] != expected_numbers:
                raise ValueError(
                    f"{video_id} {channel} indices incomplete or duplicated: "
                    f"{len(row['frames'][channel])}/{EXPECTED_FRAMES}"
                )
        if row["status_hash"] is None:
            raise ValueError(f"{video_id} missing status.json")
        if set(row["selected_frames"]) != set(selected_frame_numbers):
            raise ValueError(f"{video_id} missing one or more uniformly sampled RGB frames")
        rgb_sequence_hash = rgb_sequence_sha256(row["rgb_frame_hashes"])

        frames = np.stack([row["selected_frames"][number] for number in selected_frame_numbers])
        normalized_frames = _preprocess(frames, tf)
        video = jnp.asarray(normalized_frames[None, ...])
        forward_started = time.perf_counter()
        result = encode(video)
        jax.tree_util.tree_map(
            lambda value: value.block_until_ready()
            if hasattr(value, "block_until_ready")
            else value,
            result,
        )
        forward_seconds.append(time.perf_counter() - forward_started)
        tokens = result[0] if isinstance(result, tuple) else result
        token_array = np.asarray(tokens)
        per_frame = pool_spatial_tokens_per_frame(token_array, args.frame_count).squeeze(0)
        if per_frame.shape != (args.frame_count, 768) or not np.isfinite(per_frame).all():
            raise ValueError(f"{video_id} produced invalid per-frame features: {per_frame.shape}")

        sample_ids.append(video_id)
        status_hashes.append(row["status_hash"])
        media_hashes.append(rgb_sequence_hash)
        sampled_media_hashes.append(
            hashlib.sha256(np.ascontiguousarray(normalized_frames).tobytes()).hexdigest()
        )
        feature_rows.append(per_frame.astype(np.float32))

    with tarfile.open(archive_path, mode="r|gz") as archive:
        for member in archive:
            if not member.isfile():
                if not member.isdir():
                    raise ValueError(f"unsupported non-file tar member: {member.name}")
                parse_train_member(member.name.rstrip("/"))
                continue
            video_id, channel, frame_number = parse_train_member(member.name)
            if active_video_id != video_id:
                if active_video_id is not None and active is not None:
                    finish_video(active_video_id, active)
                if video_id in seen_video_ids:
                    raise ValueError(f"video members are not contiguous in archive: {video_id}")
                seen_video_ids.add(video_id)
                active_video_id = video_id
                active = {
                    "frames": {name: set() for name in ("scene", "depth", "masks")},
                    "selected_frames": {},
                    "rgb_frame_hashes": {},
                    "status_hash": None,
                }
            if active is None:
                raise RuntimeError("archive parser did not initialize the active video")
            if channel == "status":
                if active["status_hash"] is not None:
                    raise ValueError(f"duplicate status metadata: {member.name}")
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise ValueError(f"cannot read status metadata: {member.name}")
                try:
                    status = json.load(extracted, parse_constant=_reject_nonfinite_json_constant)
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise ValueError(f"invalid status metadata: {member.name}") from error
                active["status_hash"] = canonical_status_hash(status)
                continue
            if channel not in {"scene", "depth", "masks"} or frame_number is None:
                raise ValueError(f"unexpected train data file: {member.name}")
            if frame_number not in range(1, EXPECTED_FRAMES + 1):
                raise ValueError(f"frame index out of range: {member.name}")
            if frame_number in active["frames"][channel]:
                raise ValueError(f"duplicate frame: {member.name}")
            active["frames"][channel].add(frame_number)
            channel_counts[channel] += 1
            if channel == "scene":
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise ValueError(f"cannot read scene frame: {member.name}")
                frame_bytes = extracted.read()
                active["rgb_frame_hashes"][frame_number] = hashlib.sha256(
                    frame_bytes
                ).hexdigest()
                if frame_number in selected_frame_numbers:
                    decoded = tf.io.decode_png(frame_bytes, channels=3)
                    active["selected_frames"][frame_number] = np.asarray(decoded)
    if active_video_id is not None and active is not None:
        finish_video(active_video_id, active)
    process_memory_after = process.memory_info()
    peak_working_set = getattr(process_memory_after, "peak_wset", process_memory_after.rss)

    if not feature_rows:
        raise ValueError("training shard contains no videos")
    features = np.stack(feature_rows)
    reference_split_hashes: dict[str, str] = {}
    if args.reference_train_features:
        train_indices, validation_indices, group_ids, reference_split_hashes = (
            apply_reference_split(
                sample_ids,
                args.reference_train_features.resolve(strict=True),
                args.reference_validation_features.resolve(strict=True),
            )
        )
    else:
        train_indices, validation_indices, group_ids = deterministic_group_split(
            sample_ids,
            status_hashes,
            media_hashes,
            sampled_media_hashes=sampled_media_hashes,
            seed=args.seed,
        )
    if set(train_indices) & set(validation_indices):
        raise RuntimeError("index split overlap")
    train_groups = {group_ids[index] for index in train_indices}
    validation_groups = {group_ids[index] for index in validation_indices}
    if train_groups & validation_groups:
        raise RuntimeError("scene/media group overlap in generated split")
    train_status_ids = {status_hashes[index] for index in train_indices}
    validation_status_ids = {status_hashes[index] for index in validation_indices}
    train_media_ids = {media_hashes[index] for index in train_indices}
    validation_media_ids = {media_hashes[index] for index in validation_indices}
    train_sampled_media_ids = {sampled_media_hashes[index] for index in train_indices}
    validation_sampled_media_ids = {sampled_media_hashes[index] for index in validation_indices}
    if (
        train_status_ids & validation_status_ids
        or train_media_ids & validation_media_ids
        or train_sampled_media_ids & validation_sampled_media_ids
    ):
        raise RuntimeError("scene metadata or RGB media identities cross the split")

    output_dir.mkdir(parents=True, exist_ok=False)
    split_payloads = {
        "train": train_indices,
        "validation": validation_indices,
    }
    split_hashes: dict[str, str] = {}
    split_file_hashes: dict[str, str] = {}
    for split_name, indices in split_payloads.items():
        ids = [sample_ids[index] for index in indices]
        groups = [group_ids[index] for index in indices]
        split_hashes[split_name] = hashlib.sha256("\n".join(ids).encode()).hexdigest()
        feature_path = output_dir / f"{split_name}-features.npz"
        np.savez_compressed(
            feature_path,
            features=features[indices],
            sample_ids=np.asarray(ids),
            group_ids=np.asarray(groups),
            status_identity_sha256=np.asarray([status_hashes[index] for index in indices]),
            rgb_sequence_sha256=np.asarray([media_hashes[index] for index in indices]),
            sampled_rgb_pixels_sha256=np.asarray(
                [sampled_media_hashes[index] for index in indices]
            ),
        )
        split_file_hashes[split_name] = sha256_file(feature_path)
    metadata = {
        "status": "complete_intphys_train_frozen_feature_extraction",
        "started_at_utc": run_started_at,
        "ended_at_utc": datetime.now(UTC).isoformat(),
        "elapsed_seconds_total": time.perf_counter() - run_started,
        "archive_hashing_seconds": archive_hash_seconds,
        "model_load_seconds": model_load_seconds,
        "feature_extraction_seconds": time.perf_counter() - extraction_started,
        "command": sys.argv,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "jax_version": importlib.metadata.version("jax"),
        "tensorflow_version": module_version(tf, "tensorflow"),
        "cpu_model": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "host_ram_total_bytes": host_memory_before.total,
        "host_ram_available_at_start_bytes": host_memory_before.available,
        "process_rss_at_start_bytes": process_memory_before.rss,
        "process_rss_at_end_bytes": process_memory_after.rss,
        "process_peak_working_set_bytes": peak_working_set,
        "archive_bytes": archive_bytes,
        "archive_md5": archive_md5,
        "archive_sha256": archive_sha256,
        "archive_source": "https://download-intphys.cognitive-ml.fr/train.1.tar.gz",
        "archive_license": "CC BY 4.0",
        "archive_videos": len(sample_ids),
        "all_train_labels_possible": True,
        "video_records": len(sample_ids),
        "unique_scene_media_groups": len(set(group_ids)),
        "duplicate_status_or_media_groups": len(sample_ids) - len(set(group_ids)),
        "cross_split_group_overlap": 0,
        "cross_split_status_identity_overlap": 0,
        "cross_split_rgb_sequence_identity_overlap": 0,
        "cross_split_sampled_rgb_pixels_identity_overlap": 0,
        "channel_frame_counts": dict(channel_counts),
        "frame_count": args.frame_count,
        "selected_frame_numbers_1_based": list(selected_frame_numbers),
        "input_frame_size": FRAME_SIZE,
        "input_color_order": "RGB",
        "frames_per_video": EXPECTED_FRAMES,
        "feature_shape": list(features.shape[1:]),
        "feature_layout": "mean pooled spatial tokens per sampled frame; temporal order retained",
        "rgb_identity": "SHA-256 of ordered SHA-256 digests of all 100 original RGB PNG payloads",
        "sampled_rgb_identity": (
            f"SHA-256 of normalized {args.frame_count}-frame VideoPrism input tensor bytes"
        ),
        "model_revision": MODEL_REVISION,
        "videoprism_source_commit": source_commit,
        "checkpoint_sha256": checkpoint_sha256,
        "backend": "JAX CPU",
        "forward_seconds_median": float(np.median(forward_seconds)),
        "forward_seconds_p95": float(np.quantile(forward_seconds, 0.95)),
        "split_seed": args.seed,
        "split_policy": (
            "reused exact IDs, order, and group assignment from reference feature caches"
            if reference_split_hashes
            else "SHA-256(group identity) modulo 5; bucket 0 held for internal validation"
        ),
        "reference_split_feature_sha256": reference_split_hashes,
        "sample_id_order_sha256": split_hashes,
        "feature_archive_sha256": split_file_hashes,
        "train_examples": len(train_indices),
        "validation_examples": len(validation_indices),
        "train_unique_groups": len(train_groups),
        "validation_unique_groups": len(validation_groups),
        "development_labels_loaded": False,
        "sealed_test_accessed": False,
    }
    (output_dir / "extract-report.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
