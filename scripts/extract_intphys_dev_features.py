"""Extract one pinned VideoPrism feature sequence per IntPhys dev video."""

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
from collections import Counter, defaultdict
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
from scripts.extract_intphys_train_features import module_version

EXPECTED_ARCHIVE_BYTES = 3_203_229_667
EXPECTED_ARCHIVE_MD5 = "6a72715007f2f3b0e9546bfa8d8fc39b"
EXPECTED_ARCHIVE_SHA256 = "441eebe09f6583a18dbcd9d2b6c6b5667c66bac99e3875622e0768dc22e16a0a"
EXPECTED_FRAMES = 100
SELECTED_FRAME_NUMBERS = (1, 15, 29, 43, 57, 71, 85, 100)
_FRAME = re.compile(r"^(scene|depth|masks)_(\d+)\.png$")


def parse_dev_member(name: str) -> tuple[str, str, str, int | None]:
    """Return block, scene, rendition and frame number for known dev members."""
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != "dev":
        raise ValueError(f"unsafe or unexpected IntPhys dev path: {name}")
    if len(path.parts) == 1:
        return "", "", "", None
    if len(path.parts) == 2:
        block = path.parts[1]
        if block not in {"O1", "O2", "O3"}:
            raise ValueError(f"unexpected IntPhys block: {block}")
        return block, "", "", None
    if len(path.parts) < 3:
        raise ValueError(f"incomplete IntPhys dev path: {name}")
    block, scene = path.parts[1:3]
    if block not in {"O1", "O2", "O3"}:
        raise ValueError(f"unexpected IntPhys block: {block}")
    if len(path.parts) == 3:
        return block, scene, "", None
    if len(path.parts) == 4:
        return block, scene, "", None
    if len(path.parts) == 5 and path.name == "status.json":
        return block, scene, "status", None
    if len(path.parts) == 5 and path.parts[4] in {"scene", "depth", "masks"}:
        return block, scene, "", None
    if len(path.parts) == 6 and path.parts[4] in {"scene", "depth", "masks"}:
        match = _FRAME.fullmatch(path.name)
        if match is None or match.group(1) != path.parts[4]:
            raise ValueError(f"unexpected IntPhys frame member: {name}")
        return block, scene, match.group(1), int(match.group(2))
    raise ValueError(f"unexpected IntPhys dev member: {name}")


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


def _reject_nonfinite_json_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON number: {value}")


def validate_audit_summary(audit: dict[str, Any]) -> None:
    if (
        audit.get("archive_bytes") != EXPECTED_ARCHIVE_BYTES
        or audit.get("archive_md5") != EXPECTED_ARCHIVE_MD5
        or audit.get("archive_sha256") != EXPECTED_ARCHIVE_SHA256
        or audit.get("video_count") != 360
        or audit.get("scene_group_count") != 90
        or audit.get("matched_groups_have_two_labels_each") is not True
    ):
        raise ValueError("pinned IntPhys dev audit report differs from the admitted audit")


def sampled_media_label_collision_audit(
    group_ids: list[str], labels: list[int], media_hashes: list[str]
) -> dict[str, Any]:
    """Report exact sampled-pixel collisions across opposite labels without fitting."""
    if not group_ids or len(group_ids) != len(labels) or len(labels) != len(media_hashes):
        raise ValueError("group, label, and sampled-media arrays must have equal nonzero length")
    if any(label not in {0, 1} for label in labels):
        raise ValueError("labels must be binary")
    group_signatures: dict[str, dict[int, set[str]]] = defaultdict(
        lambda: {0: set(), 1: set()}
    )
    global_signature_labels: dict[str, set[int]] = defaultdict(set)
    for group_id, label, media_hash in zip(group_ids, labels, media_hashes, strict=True):
        if not group_id or not media_hash:
            raise ValueError("group IDs and sampled-media hashes must be nonempty")
        group_signatures[group_id][label].add(media_hash)
        global_signature_labels[media_hash].add(label)
    within_group = sorted(
        group_id
        for group_id, signatures in group_signatures.items()
        if signatures[0] & signatures[1]
    )
    return {
        "scene_groups": len(group_signatures),
        "groups_with_opposite_label_identical_sampled_media": len(within_group),
        "opposite_label_identical_sampled_media_signatures_global": sum(
            labels_seen == {0, 1} for labels_seen in global_signature_labels.values()
        ),
        "within_group_collision_group_ids": within_group,
        "interpretation": (
            "Exact collisions mean the sampled RGB inputs alone cannot distinguish those labels; "
            "zero collisions does not establish task solvability."
        ),
    }


def new_dev_video_record() -> dict[str, Any]:
    return {
        "frames": {channel: set() for channel in ("scene", "depth", "masks")},
        "selected_png": {},
        "label": None,
    }


def get_or_create_dev_video_record(
    records: dict[tuple[str, str, str], dict[str, Any]],
    record_order: list[tuple[str, str, str]],
    key: tuple[str, str, str],
) -> dict[str, Any]:
    if key not in records:
        records[key] = new_dev_video_record()
        record_order.append(key)
    return records[key]


def validate_dev_video_record(
    key: tuple[str, str, str],
    row: dict[str, Any],
    selected_frame_numbers: tuple[int, ...] = SELECTED_FRAME_NUMBERS,
) -> None:
    expected = set(range(1, EXPECTED_FRAMES + 1))
    for channel in ("scene", "depth", "masks"):
        if row["frames"][channel] != expected:
            raise ValueError(f"{key} has {len(row['frames'][channel])}/100 {channel} frames")
    if row["label"] not in {0, 1}:
        raise ValueError(f"{key} lacks a binary physical-plausibility label")
    if set(row["selected_png"]) != set(selected_frame_numbers):
        raise ValueError(f"{key} lacks a selected RGB frame")


def validate_reference_dev_rows(
    reference_path: Path,
    sample_ids: list[str],
    group_ids: list[str],
    blocks: list[str],
    labels: list[int],
) -> str:
    """Require exact ID/order/group/label parity with the prior fixed dev sample."""
    with np.load(reference_path, allow_pickle=False) as reference:
        required = {"sample_ids", "group_ids", "labels", "blocks"}
        if not required.issubset(reference.files):
            raise ValueError("reference dev cache lacks ID/group/label/block arrays")
        reference_ids = reference["sample_ids"].astype(str).tolist()
        reference_groups = reference["group_ids"].astype(str).tolist()
        reference_labels = np.asarray(reference["labels"], dtype=np.int8).tolist()
        reference_blocks = reference["blocks"].astype(str).tolist()
    if (sample_ids, group_ids, labels, blocks) != (
        reference_ids, reference_groups, reference_labels, reference_blocks
    ):
        raise ValueError("dev IDs/order/groups/labels differ from frozen reference cache")
    digest = hashlib.sha256()
    with reference_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-count", type=int, default=FRAME_COUNT)
    parser.add_argument("--reference-dev-features", type=Path)
    return parser.parse_args()


def main() -> None:
    run_started_at = datetime.now(UTC).isoformat()
    run_started = time.perf_counter()
    args = _args()
    if not 2 <= args.frame_count <= EXPECTED_FRAMES:
        raise ValueError(f"frame count must be between 2 and {EXPECTED_FRAMES}")
    selected_frame_numbers = uniform_frame_numbers(EXPECTED_FRAMES, args.frame_count)
    archive_path = args.archive.resolve(strict=True)
    audit_path = args.audit_report.resolve(strict=True)
    checkpoint = args.checkpoint.resolve(strict=True)
    source_root = args.source_root.resolve(strict=True)
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output_dir}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    validate_audit_summary(audit)
    if archive_path.stat().st_size != EXPECTED_ARCHIVE_BYTES:
        raise ValueError("IntPhys dev archive byte size differs from the audited archive")
    hash_started = time.perf_counter()
    archive_md5, archive_sha256, archive_bytes = file_digests(archive_path)
    archive_hash_seconds = time.perf_counter() - hash_started
    if (archive_md5, archive_sha256, archive_bytes) != (
        EXPECTED_ARCHIVE_MD5,
        EXPECTED_ARCHIVE_SHA256,
        EXPECTED_ARCHIVE_BYTES,
    ):
        raise ValueError("IntPhys dev archive hashes differ from the previous full audit")
    checkpoint_sha256 = hashlib.sha256()
    with checkpoint.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checkpoint_sha256.update(chunk)
    if checkpoint_sha256.hexdigest() != CHECKPOINT_SHA256:
        raise ValueError("VideoPrism checkpoint does not match the pinned R1 checkpoint")
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
        raise RuntimeError(f"CPU-only dev feature extraction required, found: {devices}")
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

    ids: list[str] = []
    group_ids: list[str] = []
    blocks: list[str] = []
    labels: list[int] = []
    features: list[np.ndarray] = []
    normalized_media_hashes: list[str] = []
    channel_counts: Counter[str] = Counter()
    forward_seconds: list[float] = []
    seen_ids: set[str] = set()
    records: dict[tuple[str, str, str], dict[str, Any]] = {}
    record_order: list[tuple[str, str, str]] = []
    extraction_started = time.perf_counter()

    def finish_video(key: tuple[str, str, str], row: dict[str, Any]) -> None:
        block, scene, rendition = key
        validate_dev_video_record(key, row, selected_frame_numbers)
        frame_array = np.stack(
            [
                np.asarray(tf.io.decode_png(row["selected_png"][number], channels=3))
                for number in selected_frame_numbers
            ]
        )
        normalized = _preprocess(frame_array, tf)
        video = jnp.asarray(normalized[None, ...])
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
            raise ValueError(f"{key} produced invalid frozen features: {per_frame.shape}")
        sample_id = f"{block}/{scene}/{rendition}"
        if sample_id in seen_ids:
            raise ValueError(f"duplicate IntPhys dev sample ID: {sample_id}")
        seen_ids.add(sample_id)
        ids.append(sample_id)
        group_ids.append(f"{block}/{scene}")
        blocks.append(block)
        labels.append(row["label"])
        features.append(per_frame.astype(np.float32))
        normalized_media_hashes.append(
            hashlib.sha256(np.ascontiguousarray(normalized).tobytes()).hexdigest()
        )

    with tarfile.open(archive_path, mode="r|gz") as archive:
        for member in archive:
            if not member.isfile():
                if not member.isdir():
                    raise ValueError(f"unsupported non-file archive member: {member.name}")
                parse_dev_member(member.name.rstrip("/"))
                continue
            block, scene, channel, frame_number = parse_dev_member(member.name)
            if channel == "":
                raise ValueError(f"unexpected data file: {member.name}")
            if channel == "status":
                key = (block, scene, PurePosixPath(member.name).parts[3])
            else:
                rendition = PurePosixPath(member.name).parts[3]
                key = (block, scene, rendition)
            active = get_or_create_dev_video_record(records, record_order, key)
            if channel == "status":
                if active["label"] is not None:
                    raise ValueError(f"duplicate status.json: {member.name}")
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise ValueError(f"cannot read status.json: {member.name}")
                status = json.load(extracted, parse_constant=_reject_nonfinite_json_constant)
                label = status.get("header", {}).get("is_possible")
                if not isinstance(label, bool):
                    raise ValueError(f"status lacks boolean is_possible: {member.name}")
                active["label"] = int(label)
                continue
            if channel not in {"scene", "depth", "masks"} or frame_number is None:
                raise ValueError(f"unexpected media member: {member.name}")
            if frame_number not in range(1, EXPECTED_FRAMES + 1):
                raise ValueError(f"frame number outside 1..100: {member.name}")
            channel_counts[channel] += 1
            if frame_number in active["frames"][channel]:
                raise ValueError(f"duplicate {channel} frame: {member.name}")
            active["frames"][channel].add(frame_number)
            if channel == "scene":
                if frame_number in selected_frame_numbers:
                    extracted = archive.extractfile(member)
                    if extracted is None:
                        raise ValueError(f"cannot read selected frame: {member.name}")
                    active["selected_png"][frame_number] = extracted.read()
    for key in record_order:
        finish_video(key, records[key])

    feature_matrix = np.stack(features)
    group_rows: dict[str, list[int]] = defaultdict(list)
    for index, group_id in enumerate(group_ids):
        group_rows[group_id].append(index)
    for group_id, indices in group_rows.items():
        group_labels = [labels[index] for index in indices]
        if len(indices) != 4 or sum(group_labels) != 2:
            raise ValueError(f"dev group {group_id} is not the audited 2/2 quadruplet")
    sampled_media_audit = sampled_media_label_collision_audit(
        group_ids, labels, normalized_media_hashes
    )
    reference_dev_hash = None
    if args.reference_dev_features:
        reference_path = args.reference_dev_features.resolve(strict=True)
        reference_dev_hash = validate_reference_dev_rows(
            reference_path, ids, group_ids, blocks, labels
        )
    process_memory_after = process.memory_info()
    peak_working_set = getattr(process_memory_after, "peak_wset", process_memory_after.rss)

    output_dir.mkdir(parents=True, exist_ok=False)
    features_path = output_dir / "dev-features.npz"
    np.savez_compressed(
        features_path,
        features=feature_matrix,
        sample_ids=np.asarray(ids),
        group_ids=np.asarray(group_ids),
        blocks=np.asarray(blocks),
        labels=np.asarray(labels, dtype=np.int8),
        normalized_media_sha256=np.asarray(normalized_media_hashes),
    )
    metadata = {
        "status": "complete_intphys_dev_frozen_features_for_one_shot_diagnostic",
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
        "audit_report_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        "channel_frame_counts": dict(channel_counts),
        "videos": len(ids),
        "scene_groups": len(group_rows),
        "sampled_media_label_collision_audit": sampled_media_audit,
        "labels_loaded_for_fit_or_selection": False,
        "development_label_use": (
            "repeated fixed development diagnostic after candidate freeze; no fitting or "
            "calibration; not blind evidence"
        ),
        "sealed_test_accessed": False,
        "frame_count": args.frame_count,
        "selected_frame_numbers_1_based": list(selected_frame_numbers),
        "input_frame_size": FRAME_SIZE,
        "feature_shape": list(feature_matrix.shape[1:]),
        "feature_layout": "mean pooled spatial tokens per sampled frame; temporal order retained",
        "model_revision": MODEL_REVISION,
        "videoprism_source_commit": source_commit,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "backend": "JAX CPU",
        "forward_seconds_median": float(np.median(forward_seconds)),
        "forward_seconds_p95": float(np.quantile(forward_seconds, 0.95)),
        "feature_file_sha256": hashlib.sha256(features_path.read_bytes()).hexdigest(),
        "reference_dev_features_sha256": reference_dev_hash,
    }
    (output_dir / "extract-report.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
