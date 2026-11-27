"""Download the frozen Physical Concepts subset and extract pinned CPU features."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

SOURCE_PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src"
FRAME_COUNT = 8
FRAME_SIZE = 288
MODEL_REVISION = "be719a406d563b66f0ac969e7c94bab8e997c81a"
SOURCE_COMMIT = "c6eba0ece3dd49a24ffc44582966997a4a59f1d8"
CHECKPOINT_SHA256 = "61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29"

EXPECTED_CATEGORIES = (
    "continuity",
    "solidity",
    "object_persistence",
    "unchangeableness",
    "directional_inertia",
)
SAMPLE_PREFIX = "PhysicalConceptsProbe-2026-11-27-v1|seed=17|"
ROW_BYTES = 15 * 64 * 64 * 3
MASK_BYTES = 2 * 15 * 64 * 64
MAX_PROBE_RECORD_BYTES = 4 * ROW_BYTES + 4 * MASK_BYTES + 64_000


def sample_rank(category: str, serialized_record: bytes) -> tuple[str, str]:
    row_sha256 = hashlib.sha256(serialized_record).hexdigest()
    rank = hashlib.sha256(f"{SAMPLE_PREFIX}{category}|{row_sha256}".encode()).hexdigest()
    return rank, row_sha256


def select_probe_records(
    records: list[tuple[str, str, str, int]], quota: int
) -> list[tuple[str, str, str, int]]:
    """Select the lowest predeclared hash ranks without label or score input."""
    if quota < 1 or len(records) < quota:
        raise ValueError("quota must be positive and no larger than the record count")
    if len({row_sha for _, row_sha, _, _ in records}) != len(records):
        raise ValueError("duplicate serialized probe records detected")
    return sorted(records)[:quota]


def media_identity_overlaps(
    selected_clip_hashes: list[str], freeform_rgb_hashes: dict[str, set[str]]
) -> dict[str, set[str]]:
    selected = set(selected_clip_hashes)
    return {split_name: selected & hashes for split_name, hashes in freeform_rgb_hashes.items()}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--encoder-checkpoint", type=Path, required=True)
    parser.add_argument("--encoder-source-root", type=Path, required=True)
    parser.add_argument("--pre-probe-validation-threshold", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reuse-pinned-payload-from", type=Path)
    parser.add_argument("--minimum-available-memory-mib", type=int, default=768)
    parser.add_argument("--progress-every", type=int, default=50)
    return parser.parse_args()


def _download_pinned_object(item: dict[str, Any], destination: Path) -> dict[str, Any]:
    object_name = item["name"]
    encoded_name = urllib.parse.quote(object_name, safe="")
    url = (
        "https://storage.googleapis.com/download/storage/v1/b/physical_concepts/o/"
        f"{encoded_name}?alt=media&generation={item['generation']}"
    )
    temporary = destination.with_suffix(destination.suffix + ".part")
    md5 = hashlib.md5()
    sha256 = hashlib.sha256()
    size = 0
    request = urllib.request.Request(url, headers={"User-Agent": "tiny-omni-decision-R2"})
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as target:
        generation = response.headers.get("x-goog-generation")
        if generation is not None and generation != item["generation"]:
            raise ValueError(f"GCS generation mismatch for {object_name}: {generation}")
        while chunk := response.read(8 * 1024 * 1024):
            target.write(chunk)
            md5.update(chunk)
            sha256.update(chunk)
            size += len(chunk)
    expected_md5 = base64.b64decode(item["md5Hash"], validate=True)
    if size != item["size"] or md5.digest() != expected_md5:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"downloaded object fails size/MD5 pin: {object_name}")
    temporary.replace(destination)
    return {
        "size": size,
        "md5_base64": base64.b64encode(md5.digest()).decode(),
        "sha256": sha256.hexdigest(),
    }


def _verify_existing_pinned_object(item: dict[str, Any], path: Path) -> dict[str, Any]:
    md5 = hashlib.md5()
    sha256 = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            md5.update(chunk)
            sha256.update(chunk)
            size += len(chunk)
    if size != item["size"] or md5.digest() != base64.b64decode(item["md5Hash"], validate=True):
        raise ValueError(
            f"existing pinned probe object fails size/MD5 verification: {item['name']}"
        )
    return {
        "size": size,
        "md5_base64": base64.b64encode(md5.digest()).decode(),
        "sha256": sha256.hexdigest(),
        "reused_from_existing_attempt": True,
    }


def _validate_probe_row(
    serialized: bytes, category: str, np: Any
) -> tuple[Any, list[int], list[str]]:
    if str(SOURCE_PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(SOURCE_PACKAGE_ROOT))
    from scripts.audit_physical_concepts_tfrecord import _feature_map

    features = _feature_map(serialized)
    expected_keys = {
        "possible_image",
        "possible_mask",
        "possible_camera_pose",
        "impossible_image",
        "impossible_mask",
        "impossible_camera_pose",
    }
    if set(features) != expected_keys:
        raise ValueError(f"unexpected {category} probe keys: {sorted(features)}")
    videos: list[Any] = []
    labels: list[int] = []
    rgb_sequence_hashes: list[str] = []
    for prefix, label in (("possible", 0), ("impossible", 1)):
        image_kind, images = features[f"{prefix}_image"]
        mask_kind, masks = features[f"{prefix}_mask"]
        pose_kind, poses = features[f"{prefix}_camera_pose"]
        if (
            image_kind != "bytes"
            or mask_kind != "bytes"
            or pose_kind != "float"
            or len(images) != 1
            or len(masks) != 1
            or len(images[0]) != 2 * ROW_BYTES
            or len(masks[0]) != MASK_BYTES
            or len(poses) != 2 * 15 * 6
            or not np.isfinite(poses).all()
        ):
            raise ValueError(f"invalid {prefix} data in {category} probe row")
        pair = np.frombuffer(images[0], dtype=np.uint8).reshape(2, 15, 64, 64, 3)
        rgb_sequence_hashes.extend(
            hashlib.sha256(np.ascontiguousarray(clip).tobytes()).hexdigest() for clip in pair
        )
        videos.extend((pair[0], pair[1]))
        labels.extend((label, label))
    return np.stack(videos), labels, rgb_sequence_hashes


def _preprocess_probe_frames(frames: Any, tf: Any, np: Any) -> Any:
    if frames.shape != (4, FRAME_COUNT, 64, 64, 3):
        raise ValueError(f"expected [4,8,64,64,3] sampled frames, got {frames.shape}")
    flat = frames.reshape(4 * FRAME_COUNT, 64, 64, 3)
    resized = tf.image.resize(flat, (FRAME_SIZE, FRAME_SIZE), method="bilinear")
    cropped = tf.image.resize_with_crop_or_pad(resized, FRAME_SIZE, FRAME_SIZE)
    output = np.asarray(cropped, dtype=np.float32).reshape(
        4, FRAME_COUNT, FRAME_SIZE, FRAME_SIZE, 3
    )
    output = output / np.float32(255.0)
    if not np.isfinite(output).all():
        raise ValueError("preprocessed probe frames contain non-finite values")
    return output


def main() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    if str(SOURCE_PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(SOURCE_PACKAGE_ROOT))
    import tensorflow as tf

    from scripts.audit_physical_concepts_tfrecord import iter_tfrecord_examples_tensorflow
    from scripts.smoke_physical_concepts_videoprism import (
        sha256_file,
        uniform_frame_indices,
    )

    args = _args()
    run_started_at = datetime.now(UTC)
    run_started = time.perf_counter()
    if args.minimum_available_memory_mib < 256 or args.progress_every < 1:
        raise ValueError("memory floor must be >=256 MiB and progress interval positive")
    manifest_path = args.manifest.resolve(strict=True)
    checkpoint = args.encoder_checkpoint.resolve(strict=True)
    source_root = args.encoder_source_root.resolve(strict=True)
    threshold_receipt_path = args.pre_probe_validation_threshold.resolve(strict=True)
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "FROZEN_BEFORE_PROBE_PAYLOAD_READ"
        or manifest.get("source", {}).get("probe_files_opened_before_freeze") is not False
        or manifest.get("candidate", {}).get("predictor_checkpoint_sha256")
        != "40e9fd86aeb384f30459c346e3a5288cc6a7281ccf1e68854b9001bf44af8e49"
    ):
        raise ValueError("probe sample/scoring protocol is not frozen for this selected checkpoint")
    if sha256_file(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("pinned VideoPrism checkpoint hash mismatch")
    candidate = manifest["candidate"]
    threshold_receipt = json.loads(threshold_receipt_path.read_text(encoding="utf-8"))
    if (
        threshold_receipt.get("status") != "frozen_before_probe_payload_download"
        or threshold_receipt.get("predictor_checkpoint_sha256")
        != candidate["predictor_checkpoint_sha256"]
        or threshold_receipt.get("predictor_metadata_sha256")
        != candidate["predictor_run_metadata_sha256"]
        or threshold_receipt.get("validation_feature_archive_sha256")
        != candidate["validation_feature_archive_sha256"]
        or abs(
            float(threshold_receipt.get("validation_mean_cosine_loss", float("inf")))
            - float(candidate["validation_cosine_loss"])
        )
        > 1e-7
        or not isinstance(threshold_receipt.get("validation_q95_anomaly_threshold"), (int, float))
        or not isinstance(threshold_receipt.get("validation_mad_scale"), (int, float))
        or threshold_receipt["validation_mad_scale"] <= 0
    ):
        raise ValueError("validation threshold receipt differs from the frozen Path A candidate")
    split_manifest_path = (
        Path(__file__).resolve().parents[1]
        / "manifests/pretrained_reuse/physical-concepts-freeform-two-shard-split-20261127.json"
    )
    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    if split_manifest.get("status") != "PROVISIONAL_TRAIN_VALIDATION_SPLIT_EXACT_OVERLAP_CLEAR":
        raise ValueError("freeform train/validation split receipt is not clean")
    freeform_record_hashes: dict[str, set[str]] = {}
    freeform_rgb_hashes: dict[str, set[str]] = {}
    freeform_audit_hashes: dict[str, str] = {}
    for split_name in ("train", "validation"):
        split = split_manifest[split_name]
        audit_path = Path(split["local_audit_path"]).resolve(strict=True)
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if (
            audit.get("status") != "complete"
            or audit.get("object_name") != split["object_name"]
            or audit.get("generation") != split["generation"]
            or audit.get("source_sha256") != split["sha256"]
            or audit.get("records") != split["records"]
        ):
            raise ValueError(f"freeform {split_name} audit does not match frozen split manifest")
        row_hashes = audit.get("record_sha256")
        rgb_hashes = audit.get("rgb_sequence_sha256")
        if not isinstance(row_hashes, list) or len(row_hashes) != split["records"]:
            raise ValueError(f"freeform {split_name} row hash inventory is incomplete")
        if not isinstance(rgb_hashes, list) or len(rgb_hashes) != split["records"]:
            raise ValueError(f"freeform {split_name} RGB identity inventory is incomplete")
        freeform_record_hashes[split_name] = set(row_hashes)
        freeform_rgb_hashes[split_name] = set(rgb_hashes)
        freeform_audit_hashes[split_name] = sha256_file(audit_path)
    if freeform_record_hashes["train"] & freeform_record_hashes["validation"]:
        raise ValueError("freeform train/validation source rows overlap")
    upstream = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if upstream != SOURCE_COMMIT:
        raise ValueError(f"unexpected VideoPrism upstream revision: {upstream}")

    objects = manifest.get("expected_source_objects", [])
    if len(objects) != 100 or {obj["category"] for obj in objects} != set(EXPECTED_CATEGORIES):
        raise ValueError(
            "pinned manifest must contain exactly 20 shards for each of five categories"
        )
    for category in EXPECTED_CATEGORIES:
        category_objects = [obj for obj in objects if obj["category"] == category]
        if len(category_objects) != 20 or len({item["name"] for item in category_objects}) != 20:
            raise ValueError(f"invalid pinned shard list for category {category}")

    output_dir.mkdir(parents=True, exist_ok=False)
    output_threshold_receipt = output_dir / "pre-probe-validation-threshold.json"
    shutil.copy2(threshold_receipt_path, output_threshold_receipt)
    reuse_root = (
        args.reuse_pinned_payload_from.resolve(strict=True) / "payload"
        if args.reuse_pinned_payload_from is not None
        else None
    )
    payload_root = output_dir / "payload"
    if reuse_root is None:
        payload_root.mkdir()
    object_receipts: dict[str, dict[str, Any]] = {}
    object_paths: dict[str, Path] = {}
    for item in objects:
        source_root = reuse_root if reuse_root is not None else payload_root
        destination = source_root / item["category"] / Path(item["name"]).name
        if reuse_root is None:
            destination.parent.mkdir(parents=True, exist_ok=True)
            object_receipts[item["name"]] = _download_pinned_object(item, destination)
        else:
            object_receipts[item["name"]] = _verify_existing_pinned_object(item, destination)
        object_paths[item["name"]] = destination
        if len(object_receipts) % 10 == 0:
            action = (
                "reused and reverified" if reuse_root is not None else "downloaded and verified"
            )
            print(f"{action} {len(object_receipts)}/100 probe shards", flush=True)

    quotas: dict[str, int] = {
        category: int(manifest["sample"]["selected_per_category"])
        for category in EXPECTED_CATEGORIES
    }
    ranked: dict[str, list[tuple[str, str, str, int]]] = defaultdict(list)
    category_counts: dict[str, dict[str, int]] = {}
    cross_category_row_hashes: dict[str, set[str]] = {
        category: set() for category in EXPECTED_CATEGORIES
    }
    for category in EXPECTED_CATEGORIES:
        count_by_shard: dict[str, int] = {}
        for item in sorted(
            (obj for obj in objects if obj["category"] == category), key=lambda obj: obj["name"]
        ):
            shard_path = object_paths[item["name"]]
            count = 0
            for index, serialized in enumerate(
                iter_tfrecord_examples_tensorflow(
                    shard_path, max_record_bytes=MAX_PROBE_RECORD_BYTES
                )
            ):
                rank, row_sha = sample_rank(category, serialized)
                ranked[category].append((rank, row_sha, item["name"], index))
                cross_category_row_hashes[category].add(row_sha)
                count += 1
            count_by_shard[item["name"]] = count
        if sum(count_by_shard.values()) != manifest["sample"]["population_per_category"]:
            raise ValueError(
                f"unexpected probe row count for {category}: {sum(count_by_shard.values())}"
            )
        if len(cross_category_row_hashes[category]) != sum(count_by_shard.values()):
            raise ValueError(f"duplicate serialized probe row within {category}")
        category_counts[category] = count_by_shard

    selected = {
        category: select_probe_records(ranked[category], quotas[category])
        for category in EXPECTED_CATEGORIES
    }
    all_selected_ids = [
        f"{category}:{row_sha}"
        for category in EXPECTED_CATEGORIES
        for _, row_sha, _, _ in selected[category]
    ]
    if len(set(all_selected_ids)) != len(all_selected_ids):
        raise ValueError("selected sample IDs overlap across concept categories")
    selected_lookup: dict[str, dict[int, tuple[int, str]]] = defaultdict(dict)
    selection_rows: list[dict[str, Any]] = []
    ordered_ids: list[str] = []
    ordered_categories: list[str] = []
    ordered_objects: list[str] = []
    for category in EXPECTED_CATEGORIES:
        for output_index, (rank, row_sha, object_name, row_index) in enumerate(selected[category]):
            global_index = len(ordered_ids)
            selected_lookup[object_name][row_index] = (global_index, row_sha)
            sample_id = f"{category}:{row_sha}"
            ordered_ids.append(sample_id)
            ordered_categories.append(category)
            ordered_objects.append(object_name)
            selection_rows.append(
                {
                    "category": category,
                    "sample_id": sample_id,
                    "row_sha256": row_sha,
                    "rank_sha256": rank,
                    "object_name": object_name,
                    "object_generation": next(
                        obj["generation"] for obj in objects if obj["name"] == object_name
                    ),
                    "row_index_zero_based": row_index,
                    "category_rank_zero_based": output_index,
                }
            )

    frame_indices = uniform_frame_indices()
    frame_cache_path = output_dir / "selected-sampled-rgb.npy"
    frame_cache = np.lib.format.open_memmap(
        frame_cache_path,
        mode="w+",
        dtype=np.uint8,
        shape=(len(ordered_ids), 4, FRAME_COUNT, 64, 64, 3),
    )
    selected_clip_hashes: list[list[str] | None] = [None] * len(ordered_ids)
    media_rows_processed = 0
    for item in objects:
        category = item["category"]
        row_indices = selected_lookup.get(item["name"], {})
        if not row_indices:
            continue
        shard_path = object_paths[item["name"]]
        for index, serialized in enumerate(
            iter_tfrecord_examples_tensorflow(shard_path, max_record_bytes=MAX_PROBE_RECORD_BYTES)
        ):
            selection = row_indices.get(index)
            if selection is None:
                continue
            output_index, expected_row_sha = selection
            if hashlib.sha256(serialized).hexdigest() != expected_row_sha:
                raise ValueError("selected probe row content hash differs from frozen selection")
            videos, _labels, clip_hashes = _validate_probe_row(serialized, category, np)
            frame_cache[output_index] = videos[:, frame_indices]
            selected_clip_hashes[output_index] = clip_hashes
            media_rows_processed += 1
            if media_rows_processed % args.progress_every == 0:
                print(
                    f"verified selected media identities {media_rows_processed}/{len(ordered_ids)}",
                    flush=True,
                )
    frame_cache.flush()
    del frame_cache
    if media_rows_processed != len(ordered_ids) or any(
        hashes is None for hashes in selected_clip_hashes
    ):
        raise ValueError(
            f"selected media preflight incomplete: {media_rows_processed}/{len(ordered_ids)}"
        )
    all_selected_clip_hashes = [
        clip_hash for group in selected_clip_hashes if group is not None for clip_hash in group
    ]
    media_overlap_by_split = media_identity_overlaps(all_selected_clip_hashes, freeform_rgb_hashes)
    if any(media_overlap_by_split.values()):
        raise ValueError(
            "selected Physical Concepts RGB sequence overlaps freeform train/validation: "
            + json.dumps({name: len(values) for name, values in media_overlap_by_split.items()})
        )
    for sample_index, clip_hashes in enumerate(selected_clip_hashes):
        selection_rows[sample_index]["clip_rgb_sequence_sha256"] = clip_hashes
    unique_selected_clip_hashes = set(all_selected_clip_hashes)

    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    import jax
    import jax.numpy as jnp
    import psutil
    from videoprism import models as vp

    from tiny_omni_decision.temporal_tokens import pool_spatial_tokens_per_frame

    if jax.default_backend() != "cpu" or any(device.platform != "cpu" for device in jax.devices()):
        raise RuntimeError(f"CPU-only probe extraction found non-CPU JAX devices: {jax.devices()}")
    process = psutil.Process()
    memory_floor = args.minimum_available_memory_mib * 1024 * 1024
    memory_before_model = psutil.virtual_memory().available
    if memory_before_model < memory_floor:
        raise MemoryError("available RAM below floor before VideoPrism model load")
    model_started = time.perf_counter()
    model = vp.get_model("videoprism_public_v1_base")
    variables = vp.load_pretrained_weights(
        "videoprism_public_v1_base", checkpoint_path=str(checkpoint)
    )
    encode = jax.jit(lambda video: model.apply(variables, video, train=False))
    model_load_seconds = time.perf_counter() - model_started
    memory_after_model = psutil.virtual_memory().available
    if memory_after_model < memory_floor:
        raise MemoryError("available RAM below floor after VideoPrism model load")

    feature_array = np.empty((len(ordered_ids), 4, FRAME_COUNT, 768), dtype=np.float32)
    sampled_rgb = np.load(frame_cache_path, mmap_mode="r", allow_pickle=False)
    forward_seconds: list[float] = []
    processed = 0
    for output_index in range(len(ordered_ids)):
        video = jnp.asarray(_preprocess_probe_frames(sampled_rgb[output_index], tf, np))
        if psutil.virtual_memory().available < memory_floor:
            raise MemoryError("available RAM fell below floor before probe encoder forward")
        started = time.perf_counter()
        result = encode(video)
        jax.tree_util.tree_map(
            lambda value: (
                value.block_until_ready() if hasattr(value, "block_until_ready") else value
            ),
            result,
        )
        forward_seconds.append(time.perf_counter() - started)
        tokens = result[0] if isinstance(result, tuple) else result
        per_frame = pool_spatial_tokens_per_frame(np.asarray(tokens), FRAME_COUNT)
        if per_frame.shape != (4, FRAME_COUNT, 768) or not np.isfinite(per_frame).all():
            raise ValueError(f"invalid pooled probe feature output: {per_frame.shape}")
        feature_array[output_index] = per_frame.astype(np.float32)
        processed += 1
        if processed % args.progress_every == 0:
            print(f"extracted {processed}/{len(ordered_ids)} probe quadruplets", flush=True)

    if processed != len(ordered_ids) or not np.isfinite(feature_array).all():
        raise ValueError(f"probe feature extraction incomplete: {processed}/{len(ordered_ids)}")
    feature_path = output_dir / "probe-features.npz"
    np.savez_compressed(
        feature_path,
        features=feature_array,
        sample_ids=np.asarray(ordered_ids),
        categories=np.asarray(ordered_categories),
        source_objects=np.asarray(ordered_objects),
        clip_labels=np.tile(np.asarray([0, 0, 1, 1], dtype=np.int8), len(ordered_ids)),
    )
    selection_path = output_dir / "sample-selection.json"
    selection_doc = {
        "status": "complete_label_score_blind_hash_ranked_probe_sample",
        "source_manifest_sha256": sha256_file(manifest_path),
        "pre_probe_validation_threshold_sha256": sha256_file(output_threshold_receipt),
        "sample_seed": manifest["sample"]["selection_seed"],
        "sample_quota_per_category": quotas,
        "category_row_counts_per_shard": category_counts,
        "sample_ids_order_sha256": hashlib.sha256("\n".join(ordered_ids).encode()).hexdigest(),
        "sample_rows": selection_rows,
        "exact_rgb_media_overlap_with_freeform": {
            split_name: len(hashes) for split_name, hashes in media_overlap_by_split.items()
        },
        "selected_clip_count": len(all_selected_clip_hashes),
        "unique_selected_rgb_clips": len(unique_selected_clip_hashes),
        "duplicate_selected_rgb_clip_references": len(all_selected_clip_hashes)
        - len(unique_selected_clip_hashes),
        "probe_data_opened_after_model_selection": True,
    }
    selection_path.write_text(
        json.dumps(selection_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = {
        "status": "complete_frozen_physical_concepts_probe_feature_extraction",
        "started_at_utc": run_started_at.isoformat(),
        "ended_at_utc": datetime.now(UTC).isoformat(),
        "elapsed_seconds": time.perf_counter() - run_started,
        "source_manifest_sha256": sha256_file(manifest_path),
        "freeform_split_manifest_sha256": sha256_file(split_manifest_path),
        "freeform_audit_sha256": freeform_audit_hashes,
        "sample_selection_sha256": sha256_file(selection_path),
        "pre_probe_validation_threshold_sha256": sha256_file(output_threshold_receipt),
        "source_object_receipts": object_receipts,
        "reused_pinned_payload_from": (
            str(args.reuse_pinned_payload_from.resolve(strict=True))
            if args.reuse_pinned_payload_from is not None
            else None
        ),
        "model_id": "google/videoprism-base-f16r288",
        "model_revision": MODEL_REVISION,
        "encoder_checkpoint_sha256": CHECKPOINT_SHA256,
        "encoder_source_commit": upstream,
        "backend": jax.default_backend(),
        "jax_version": jax.__version__,
        "tensorflow_version": tf.__version__,
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_compute_used": False,
        "feature_shape": list(feature_array.shape),
        "feature_values_sha256": hashlib.sha256(
            np.ascontiguousarray(feature_array).tobytes()
        ).hexdigest(),
        "feature_file_sha256": sha256_file(feature_path),
        "feature_file_bytes": feature_path.stat().st_size,
        "source_quadruplets_processed": processed,
        "videos_processed": processed * 4,
        "model_load_seconds": model_load_seconds,
        "mean_forward_seconds_per_quadruplet": float(np.mean(forward_seconds)),
        "mean_forward_seconds_per_video": float(np.sum(forward_seconds) / (processed * 4)),
        "sum_forward_seconds": float(np.sum(forward_seconds)),
        "peak_working_set_bytes": getattr(process.memory_info(), "peak_wset", None),
        "available_ram_before_model_bytes": memory_before_model,
        "available_ram_after_model_bytes": memory_after_model,
        "available_ram_after_forward_bytes": psutil.virtual_memory().available,
        "freeform_exact_row_overlap": {
            split_name: len(hashes) for split_name, hashes in media_overlap_by_split.items()
        },
        "quality_claim": "none; features only, probe metrics are computed separately once",
    }
    report_path = output_dir / "feature-extraction-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "source_object_receipts"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
