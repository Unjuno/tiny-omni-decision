"""One-shot fixed-readout transfer evaluation on untouched CLEVR-4 100k val."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from zipfile import ZipFile

import numpy as np

TAXONOMIES = ("color", "texture", "count", "shape")
EXPECTED_COUNTS = {"train": 80_012, "val": 19_988}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return sha256_bytes(payload)


def read_json_member(archive: ZipFile, member: str) -> tuple[object, str]:
    payload = archive.read(member)
    return json.loads(payload), sha256_bytes(payload)


def old_manifest_state(paths: list[Path]) -> tuple[set[str], set[str], dict[str, int]]:
    ids: set[str] = set()
    media_hashes: set[str] = set()
    counts: dict[str, int] = {}
    for path in paths:
        count = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            row = json.loads(line)
            ids.add(str(row["id"]))
            media_hashes.add(str(row["sha256"]).lower())
            count += 1
        counts[path.name] = count
    return ids, media_hashes, counts


def ece(probabilities: np.ndarray, targets: np.ndarray, bins: int = 15) -> float:
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == targets
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for index in range(bins):
        lower = (
            confidence >= boundaries[index]
            if index == 0
            else confidence > boundaries[index]
        )
        mask = lower & (confidence <= boundaries[index + 1])
        if mask.any():
            result += float(mask.mean()) * abs(
                float(correct[mask].mean()) - float(confidence[mask].mean())
            )
    return result


def metrics(probabilities: np.ndarray, targets: np.ndarray) -> dict[str, float | int]:
    rows = np.arange(len(targets))
    chosen = probabilities[rows, targets]
    one_hot = np.eye(probabilities.shape[1], dtype=np.float64)[targets]
    return {
        "count": int(len(targets)),
        "accuracy": float((probabilities.argmax(axis=1) == targets).mean()),
        "nll": float(-np.log(np.clip(chosen, 1e-12, 1.0)).mean()),
        "brier": float(((probabilities - one_hot) ** 2).sum(axis=1).mean()),
        "ece_15_equal_width": ece(probabilities, targets),
        "mean_confidence": float(probabilities.max(axis=1).mean()),
    }


def encode_png_batch(runtime: object, pngs: list[bytes]) -> np.ndarray:
    """Match the pinned runtime preprocessing while decoding from ZIP bytes."""
    if not pngs or len(pngs) > 8:
        raise ValueError("PNG batch must contain 1..8 images")
    tf = runtime.tf
    frames = []
    for payload in pngs:
        image = tf.io.decode_png(payload, channels=3)
        height, width = int(image.shape[0]), int(image.shape[1])
        scale = max(288 / height, 288 / width)
        resized = tf.image.resize(
            image, (round(height * scale), round(width * scale)), method="bilinear"
        )
        cropped = tf.image.resize_with_crop_or_pad(resized, 288, 288)
        frame = np.asarray(cropped, dtype=np.float32) / 255.0
        frames.append(np.repeat(frame[None, ...], 8, axis=0))
    while len(frames) < 8:
        frames.append(frames[-1].copy())
    video = np.stack(frames)
    result = runtime.encode(runtime.jax.numpy.asarray(video))
    runtime.jax.tree_util.tree_map(
        lambda value: value.block_until_ready() if hasattr(value, "block_until_ready") else value,
        result,
    )
    tokens = result[0] if isinstance(result, tuple) else result
    pooled = np.asarray(tokens).mean(axis=1).astype(np.float32)[: len(pngs)]
    if pooled.shape != (len(pngs), 768) or not np.isfinite(pooled).all():
        raise ValueError("VideoPrism returned invalid features")
    return pooled


def score_one(runtime: object, image_features: np.ndarray, taxonomy: str) -> np.ndarray:
    from tiny_omni_decision.videoprism_runtime import (
        CLEVR4_TAXONOMIES,
        candidate_features,
        score_candidates,
    )

    options = CLEVR4_TAXONOMIES[taxonomy]
    vectors = np.stack([runtime.option_embeddings[f"{taxonomy}__{option}"] for option in options])
    logits = score_candidates(candidate_features(image_features, vectors), runtime.weights)
    logits = logits - logits.max()
    probabilities = np.exp(logits)
    return (probabilities / probabilities.sum()).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--source-preflight", type=Path, required=True)
    parser.add_argument("--old-manifest", type=Path, action="append", required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "predictions.jsonl").touch(exist_ok=False)
    started = datetime.now(UTC)
    monotonic_start = time.monotonic()
    status_path = args.output_dir / "run-metadata.json"
    metadata: dict[str, object] = {
        "schema_version": 1,
        "status": "starting",
        "started_utc": started.isoformat(),
        "evaluation_scope": "one-shot frozen-readout transfer evaluation on official validation",
        "sealed_audit": False,
        "checkpoint_selection_or_calibration": False,
        "hyperparameter_changes": False,
        "training_performed": False,
    }
    status_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    try:
        preflight = json.loads(args.source_preflight.read_text(encoding="utf-8"))
        if (
            preflight.get("archive_sha512_match") is not True
            or preflight.get("archive_sha512_actual") != preflight.get("archive_sha512")
            or args.archive.stat().st_size != preflight.get("archive_bytes")
        ):
            raise ValueError("source preflight hash or current archive size is not verified")
        archive_sha512 = preflight["archive_sha512_actual"]

        old_ids, old_media_hashes, old_counts = old_manifest_state(args.old_manifest)
        manifest_path = args.output_dir / "validation-manifest.jsonl"
        manifest_digest = hashlib.sha256()
        split_counts: Counter[str] = Counter()
        taxonomy_values: dict[str, set[str]] = {name: set() for name in TAXONOMIES}
        rows: dict[str, dict[str, object]] = {}
        with ZipFile(args.archive) as archive:
            names = set(archive.namelist())
            annotation_obj, annotation_sha256 = read_json_member(archive, "clevr_4_annots.json")
            if not isinstance(annotation_obj, dict):
                raise TypeError("annotation payload must be an ID-keyed object")
            rows = {str(key): value for key, value in annotation_obj.items()}
            for sample_id, row in rows.items():
                if not isinstance(row, dict) or any(
                    name not in row for name in ("split", *TAXONOMIES)
                ):
                    raise ValueError(f"invalid annotation schema for {sample_id}")
                split_counts[str(row["split"])] += 1
                for taxonomy in TAXONOMIES:
                    taxonomy_values[taxonomy].add(str(row[taxonomy]))
            if dict(split_counts) != EXPECTED_COUNTS:
                raise ValueError(f"unexpected official split counts: {dict(split_counts)}")

            validation_ids = sorted(
                sample_id for sample_id, row in rows.items() if row["split"] == "val"
            )
            id_overlap = set(validation_ids) & old_ids
            if id_overlap:
                raise ValueError(f"validation overlaps old corpus IDs: {len(id_overlap)}")
            expected_vocabularies = {
                "count": set(str(value) for value in range(1, 11)),
                "color": {
                    "gray",
                    "red",
                    "blue",
                    "green",
                    "brown",
                    "purple",
                    "cyan",
                    "yellow",
                    "pink",
                    "orange",
                },
                "texture": {
                    "rubber",
                    "metal",
                    "checkered",
                    "emojis",
                    "wave",
                    "brick",
                    "star",
                    "circles",
                    "zigzag",
                    "chessboard",
                },
                "shape": {
                    "cube",
                    "sphere",
                    "monkey",
                    "cone",
                    "torus",
                    "star",
                    "teapot",
                    "diamond",
                    "gear",
                    "cylinder",
                },
            }
            for taxonomy in TAXONOMIES:
                if taxonomy_values[taxonomy] != expected_vocabularies[taxonomy]:
                    raise ValueError(
                        f"unexpected {taxonomy} taxonomy: {sorted(taxonomy_values[taxonomy])}"
                    )
            missing = [
                f"images/{sample_id}.png"
                for sample_id in validation_ids
                if f"images/{sample_id}.png" not in names
            ]
            if missing:
                raise FileNotFoundError(f"missing validation media members: {missing[:3]}")

            manifest_rows: list[dict[str, object]] = []
            media_overlap: set[str] = set()
            for sample_id in validation_ids:
                member = f"images/{sample_id}.png"
                media_hash = hashlib.sha256(archive.read(member)).hexdigest()
                if media_hash in old_media_hashes:
                    media_overlap.add(media_hash)
                row = {
                    "id": sample_id,
                    "split": "val",
                    "member": member,
                    "media_sha256": media_hash,
                    "targets": {name: str(rows[sample_id][name]) for name in TAXONOMIES},
                }
                line = json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                encoded = line.encode("utf-8")
                manifest_digest.update(encoded)
                manifest_rows.append(row)
            if media_overlap:
                raise ValueError(
                    f"validation overlaps old corpus media hashes: {len(media_overlap)}"
                )
            with manifest_path.open("x", encoding="utf-8", newline="\n") as stream:
                for row in manifest_rows:
                    stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")

            # Select the dedicated JAX/TF profile and force CPU before loading JAX.
            os.environ["JAX_PLATFORMS"] = "cpu"
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
            runtime_dir = args.bundle / "runtime"
            sys.path.insert(0, str(runtime_dir))
            from tiny_omni_decision.videoprism_runtime import VideoPrismClevr4Runtime

            runtime = VideoPrismClevr4Runtime(args.bundle, args.source_root)
            backend = runtime.jax.default_backend()
            if backend != "cpu":
                raise RuntimeError(f"expected JAX CPU backend, got {backend}")
            metadata.update(
                {
                    "status": "preflight_complete",
                    "source_archive_sha512": archive_sha512,
                    "source_archive_bytes": args.archive.stat().st_size,
                    "source_archive_last_write_utc": datetime.fromtimestamp(
                        args.archive.stat().st_mtime, UTC
                    ).isoformat(),
                    "source_archive_full_hash_reused_from_preflight": True,
                    "source_revision": preflight["dataset_code_commit"],
                    "source_license": preflight["license"],
                    "source_archive_has_test_partition": False,
                    "annotation_sha256": annotation_sha256,
                    "official_split_counts": dict(split_counts),
                    "validation_count": len(validation_ids),
                    "old_manifest_counts": old_counts,
                    "old_unique_id_count": len(old_ids),
                    "old_unique_media_hash_count": len(old_media_hashes),
                    "validation_id_overlap_count": len(id_overlap),
                    "validation_media_hash_overlap_count": len(media_overlap),
                    "validation_manifest_sha256": manifest_digest.hexdigest(),
                    "validation_id_order_sha256": canonical_json_hash(validation_ids),
                    "label_vocabularies": {k: sorted(v) for k, v in taxonomy_values.items()},
                    "bundle_manifest_sha256": sha256_file(args.bundle / "bundle-manifest.json"),
                    "bundle_id": runtime.manifest["bundle_id"],
                    "encoder_revision": runtime.manifest["encoder"]["revision"],
                    "encoder_checkpoint_sha256": runtime.manifest["encoder"]["checkpoint_sha256"],
                    "readout_sha256": runtime.manifest["readout"]["export_sha256"],
                    "readout_source_sha256": runtime.manifest["readout"]["source_sha256"],
                    "resident_parameters": runtime.manifest["total_parameters"],
                    "evaluation_backend": backend,
                    "environment": {
                        "python": sys.version,
                        "platform": platform.platform(),
                        "processor": platform.processor(),
                        "jax": runtime.jax.__version__,
                        "tensorflow": runtime.tf.__version__,
                        "numpy": np.__version__,
                        "gpu_evaluation": False,
                    },
                    "batch_size": 8,
                    "preprocessing": (
                        "pinned runtime: TF PNG decode; resize short side 288 bilinear; "
                        "center crop; repeat image over 8 frames; mean all VideoPrism tokens"
                    ),
                }
            )
            effective_config = {
                "archive": str(args.archive.resolve()),
                "archive_sha512": archive_sha512,
                "annotation_sha256": annotation_sha256,
                "old_manifests": [str(path.resolve()) for path in args.old_manifest],
                "bundle_dir": str(args.bundle.resolve()),
                "bundle_manifest_sha256": metadata["bundle_manifest_sha256"],
                "validation_count": len(validation_ids),
                "validation_manifest_sha256": manifest_digest.hexdigest(),
                "validation_id_order_sha256": canonical_json_hash(validation_ids),
                "taxonomies": list(TAXONOMIES),
                "options_per_taxonomy": 10,
                "batch_size": 8,
                "training": False,
                "checkpoint_selection": False,
                "calibration": False,
                "sealed_audit": False,
                "device": "cpu",
            }
            (args.output_dir / "effective-config.json").write_text(
                json.dumps(effective_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            metadata["effective_config_sha256"] = canonical_json_hash(effective_config)
            metadata["runner_sha256"] = sha256_file(Path(__file__))
            status_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

            probabilities_by_taxonomy: dict[str, list[np.ndarray]] = {
                name: [] for name in TAXONOMIES
            }
            targets_by_taxonomy: dict[str, list[int]] = {name: [] for name in TAXONOMIES}
            from tiny_omni_decision.videoprism_runtime import CLEVR4_TAXONOMIES

            option_indices = {
                taxonomy: {
                    option: index for index, option in enumerate(CLEVR4_TAXONOMIES[taxonomy])
                }
                for taxonomy in TAXONOMIES
            }
            predictions_path = args.output_dir / "predictions.jsonl"
            with predictions_path.open("a", encoding="utf-8", newline="\n") as output:
                for start in range(0, len(validation_ids), 8):
                    batch_ids = validation_ids[start : start + 8]
                    pngs = [archive.read(f"images/{sample_id}.png") for sample_id in batch_ids]
                    features = encode_png_batch(runtime, pngs)
                    for sample_id, feature in zip(batch_ids, features, strict=True):
                        per_taxonomy: dict[str, object] = {}
                        for taxonomy in TAXONOMIES:
                            probs = score_one(runtime, feature, taxonomy)
                            target_text = str(rows[sample_id][taxonomy])
                            target_idx = option_indices[taxonomy][target_text]
                            probabilities_by_taxonomy[taxonomy].append(probs.astype(np.float64))
                            targets_by_taxonomy[taxonomy].append(target_idx)
                            per_taxonomy[taxonomy] = {
                                "options": CLEVR4_TAXONOMIES[taxonomy],
                                "target": target_text,
                                "prediction": CLEVR4_TAXONOMIES[taxonomy][int(probs.argmax())],
                                "probabilities": [float(value) for value in probs],
                            }
                        output.write(
                            json.dumps(
                                {"id": sample_id, "tasks": per_taxonomy}, separators=(",", ":")
                            )
                            + "\n"
                        )
                    output.flush()
                    completed = min(start + len(batch_ids), len(validation_ids))
                    if completed % 64 == 0 or completed == len(validation_ids):
                        metadata["completed_images"] = completed
                        metadata["elapsed_seconds"] = time.monotonic() - monotonic_start
                        metadata["status"] = "running"
                        status_path.write_text(
                            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
                        )

        by_taxonomy = {
            taxonomy: metrics(
                np.stack(probabilities_by_taxonomy[taxonomy]),
                np.asarray(targets_by_taxonomy[taxonomy]),
            )
            for taxonomy in TAXONOMIES
        }
        all_targets = np.concatenate([np.asarray(targets_by_taxonomy[name]) for name in TAXONOMIES])
        macro = {
            key: float(np.mean([by_taxonomy[name][key] for name in TAXONOMIES]))
            for key in ("accuracy", "nll", "brier", "ece_15_equal_width", "mean_confidence")
        }
        metadata.update(
            {
                "status": "complete",
                "ended_utc": datetime.now(UTC).isoformat(),
                "elapsed_seconds": time.monotonic() - monotonic_start,
                "completed_images": len(validation_ids),
                "decision_count": int(len(all_targets)),
                "macro_metrics": macro,
                "per_taxonomy_metrics": by_taxonomy,
                "prediction_sha256": sha256_file(args.output_dir / "predictions.jsonl"),
                "evaluation_ran_once": True,
            }
        )
        status_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    except Exception as exc:
        metadata.update(
            {
                "status": "failed",
                "ended_utc": datetime.now(UTC).isoformat(),
                "elapsed_seconds": time.monotonic() - monotonic_start,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        status_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
