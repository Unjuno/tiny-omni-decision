"""Label-blind low-resolution scene-similarity screen across train/validation shards."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from scripts.audit_physical_concepts_tfrecord import _feature_map, iter_tfrecord_examples


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_receipt(path: Path) -> tuple[int, str, str]:
    md5 = hashlib.md5()
    sha256 = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            md5.update(chunk)
            sha256.update(chunk)
            size += len(chunk)
    return size, base64.b64encode(md5.digest()).decode("ascii"), sha256.hexdigest()


def _normalize_group(values: Any, np: Any) -> Any:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(flat))
    if not math.isfinite(norm):
        raise ValueError("non-finite scene descriptor")
    return flat / max(norm, 1e-12)


def scene_signature(serialized: bytes, np: Any) -> Any:
    """Describe coarse appearance, segmentation occupancy, and camera motion."""
    features = _feature_map(serialized)
    if set(features) != {"image", "mask", "camera_pose"}:
        raise ValueError("unexpected Physical Concepts feature keys")
    image_kind, image_values = features["image"]
    mask_kind, mask_values = features["mask"]
    pose_kind, pose_values = features["camera_pose"]
    if (
        image_kind != "bytes"
        or mask_kind != "bytes"
        or pose_kind != "float"
        or len(image_values) != 1
        or len(mask_values) != 1
        or len(image_values[0]) != 15 * 64 * 64 * 3
        or len(mask_values[0]) != 15 * 64 * 64
        or len(pose_values) != 90
        or not all(math.isfinite(value) for value in pose_values)
    ):
        raise ValueError("invalid Physical Concepts row schema")

    frames = np.frombuffer(image_values[0], dtype=np.uint8).reshape(15, 64, 64, 3)
    frame_grid = frames.reshape(15, 8, 8, 8, 8, 3).mean(axis=(3, 4)) / 255.0
    rgb = np.concatenate(
        [frame_grid.mean(axis=0).reshape(-1), frame_grid.std(axis=0).reshape(-1)]
    )

    masks = np.frombuffer(mask_values[0], dtype=np.uint8).reshape(15, 64, 64)
    mask_grid = masks.reshape(15, 8, 8, 8, 8).mean(axis=(3, 4)) / 255.0
    mask = np.concatenate([mask_grid.mean(axis=0).reshape(-1), mask_grid.std(axis=0).reshape(-1)])

    poses = np.asarray(pose_values, dtype=np.float32).reshape(15, 6)
    pose = np.concatenate([poses.mean(axis=0), poses.std(axis=0)])
    groups = [_normalize_group(rgb, np), _normalize_group(mask, np), _normalize_group(pose, np)]
    signature = _normalize_group(np.concatenate(groups), np).astype(np.float32)
    if not np.isfinite(signature).all():
        raise ValueError("scene signature contains non-finite values")
    return signature


def load_split_signatures(
    *, manifest_entry: dict[str, Any], audit_report: dict[str, Any], np: Any
) -> tuple[list[str], Any]:
    """Verify a local shard against its pinned receipt and compute row signatures."""
    if (
        audit_report.get("status") != "complete"
        or audit_report.get("object_name") != manifest_entry.get("object_name")
        or audit_report.get("generation") != manifest_entry.get("generation")
    ):
        raise ValueError("audit report does not match the pinned manifest shard")
    path = Path(manifest_entry["local_payload_path"]).resolve(strict=True)
    size, md5_b64, sha256 = _file_receipt(path)
    if (
        size != manifest_entry["compressed_bytes"]
        or md5_b64 != manifest_entry["md5_base64"]
        or sha256 != manifest_entry["sha256"]
        or sha256 != audit_report["source_sha256"]
    ):
        raise ValueError("local shard does not match manifest and audit hashes")
    expected_row_hashes = audit_report["record_sha256"]
    identifiers: list[str] = []
    signatures = []
    for index, row in enumerate(iter_tfrecord_examples(path)):
        row_hash = hashlib.sha256(row).hexdigest()
        if index >= len(expected_row_hashes) or row_hash != expected_row_hashes[index]:
            raise ValueError(f"record identity mismatch at row {index}")
        identifiers.append(row_hash)
        signatures.append(scene_signature(row, np))
    if len(signatures) != manifest_entry["records"] or len(signatures) != len(expected_row_hashes):
        raise ValueError("local shard record count differs from pinned audit")
    return identifiers, np.stack(signatures)


def _quantiles(values: Any, np: Any) -> dict[str, float]:
    result = np.percentile(values, [50, 90, 95, 99, 100])
    return {
        name: float(value)
        for name, value in zip(("p50", "p90", "p95", "p99", "max"), result, strict=True)
    }


def _group_nearest(train: Any, validation: Any, np: Any) -> dict[str, Any]:
    train_norm = train / np.maximum(np.linalg.norm(train, axis=1, keepdims=True), 1e-12)
    validation_norm = validation / np.maximum(
        np.linalg.norm(validation, axis=1, keepdims=True), 1e-12
    )
    similarity = train_norm @ validation_norm.T
    train_max = similarity.max(axis=1)
    validation_max = similarity.max(axis=0)
    return {
        "train_to_validation_nearest": _quantiles(train_max, np),
        "validation_to_train_nearest": _quantiles(validation_max, np),
        "nearest_counts": {
            str(threshold): {
                "train_to_validation": int((train_max >= threshold).sum()),
                "validation_to_train": int((validation_max >= threshold).sum()),
            }
            for threshold in (0.99, 0.995)
        },
    }


def screen_split(
    train_ids: list[str],
    train_signatures: Any,
    validation_ids: list[str],
    validation_signatures: Any,
    *,
    np: Any,
    top_k: int = 25,
) -> dict[str, Any]:
    """Report nearest cross-split coarse descriptors without filtering examples."""
    if not train_ids or not validation_ids or top_k < 1:
        raise ValueError("non-empty splits and positive top_k are required")
    if (
        train_signatures.ndim != 2
        or validation_signatures.ndim != 2
        or train_signatures.shape[1] != 524
        or validation_signatures.shape[1] != 524
    ):
        raise ValueError("scene signatures must have the expected 524D layout")
    train_norms = np.linalg.norm(train_signatures, axis=1)
    validation_norms = np.linalg.norm(validation_signatures, axis=1)
    if not np.allclose(train_norms, 1.0, atol=1e-5) or not np.allclose(
        validation_norms, 1.0, atol=1e-5
    ):
        raise ValueError("scene signatures must be unit normalized for cosine similarity")
    similarity = train_signatures @ validation_signatures.T
    if not np.isfinite(similarity).all():
        raise ValueError("non-finite cross-split similarities")
    train_max = similarity.max(axis=1)
    validation_max = similarity.max(axis=0)
    flat_indices = np.argpartition(similarity.reshape(-1), -min(top_k, similarity.size))[-top_k:]
    best = sorted(
        (
            {
                "similarity": float(similarity.flat[index]),
                "train_record_sha256": train_ids[index // len(validation_ids)],
                "validation_record_sha256": validation_ids[index % len(validation_ids)],
            }
            for index in flat_indices
        ),
        key=lambda item: item["similarity"],
        reverse=True,
    )
    return {
        "status": "complete_label_blind_scene_similarity_screen",
        "train_records": len(train_ids),
        "validation_records": len(validation_ids),
        "signature_dimension": int(train_signatures.shape[1]),
        "similarity": (
            "cosine of concatenated unit-normalized groups: coarse RGB appearance, "
            "mask occupancy, camera pose"
        ),
        "train_to_validation_nearest_cosine": _quantiles(train_max, np),
        "validation_to_train_nearest_cosine": _quantiles(validation_max, np),
        "feature_group_nearest_cosine": {
            "coarse_rgb_appearance": _group_nearest(
                train_signatures[:, :384], validation_signatures[:, :384], np
            ),
            "mask_occupancy": _group_nearest(
                train_signatures[:, 384:512], validation_signatures[:, 384:512], np
            ),
            "camera_pose": _group_nearest(
                train_signatures[:, 512:524], validation_signatures[:, 512:524], np
            ),
        },
        "nearest_counts": {
            str(threshold): {
                "train_to_validation": int((train_max >= threshold).sum()),
                "validation_to_train": int((validation_max >= threshold).sum()),
            }
            for threshold in (0.98, 0.99, 0.995)
        },
        "top_pairs": best,
        "examples_removed_or_reassigned": 0,
        "interpretation_limit": (
            "Heuristic low-resolution similarities only; not verified scene identity "
            "or human-reviewed duplicates."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--train-audit", type=Path, required=True)
    parser.add_argument("--validation-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    train_report = json.loads(args.train_audit.read_text(encoding="utf-8"))
    validation_report = json.loads(args.validation_audit.read_text(encoding="utf-8"))
    import numpy as np

    train_ids, train_signatures = load_split_signatures(
        manifest_entry=manifest["train"], audit_report=train_report, np=np
    )
    validation_ids, validation_signatures = load_split_signatures(
        manifest_entry=manifest["validation"], audit_report=validation_report, np=np
    )
    report = screen_split(
        train_ids, train_signatures, validation_ids, validation_signatures, np=np
    )
    report["source_manifest_sha256"] = _sha256_file(args.manifest)
    report["audit_report_sha256"] = {
        "train": _sha256_file(args.train_audit),
        "validation": _sha256_file(args.validation_audit),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing audit output: {args.output}")
    args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    print(f"receipt_sha256={hashlib.sha256(serialized.encode('utf-8')).hexdigest()}")


if __name__ == "__main__":
    main()
