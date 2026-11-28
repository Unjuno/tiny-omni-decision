"""Measure within-modality cross-point identity retrieval on frozen features."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

MODALITIES = ("image", "audio", "tactile")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rank_cross_modal(
    query_features: np.ndarray,
    gallery_features: np.ndarray,
    records: list[dict[str, Any]],
    partition: str,
    *,
    query_modality: str,
    gallery_modality: str,
):
    if (
        query_features.ndim != 2
        or gallery_features.ndim != 2
        or query_features.shape[0] != len(records)
        or gallery_features.shape[0] != len(records)
    ):
        raise ValueError("feature rows and metadata records do not align")
    if query_features.shape[1] != gallery_features.shape[1]:
        raise ValueError("cosine retrieval requires equal feature dimensions")
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(records):
        if row["partition"] == partition:
            grouped[row["object_id"]].append(index)
    point_counts = {len(indices) for indices in grouped.values()}
    if len(grouped) < 2 or len(point_counts) != 1 or next(iter(point_counts)) < 2:
        raise ValueError(
            f"{partition} needs an equal count of at least two points per object "
            "and at least 2 objects"
        )

    object_ids = sorted(grouped)
    query_points = np.stack(
        [query_features[grouped[object_id]] for object_id in object_ids]
    )
    gallery_points = np.stack(
        [gallery_features[grouped[object_id]] for object_id in object_ids]
    )
    if not np.isfinite(query_points).all() or not np.isfinite(gallery_points).all():
        raise ValueError("features contain NaN or Inf")
    query_norm = np.linalg.norm(query_points, axis=-1, keepdims=True)
    gallery_norm = np.linalg.norm(gallery_points, axis=-1, keepdims=True)
    if (query_norm <= 1e-12).any() or (gallery_norm <= 1e-12).any():
        raise ValueError("zero feature vector cannot support cosine retrieval")
    query_points = query_points / query_norm
    gallery_points = gallery_points / gallery_norm

    predictions = []
    ranks = []
    point_count = query_points.shape[1]
    for query_point in range(point_count):
        for gallery_point in range(point_count):
            if query_point == gallery_point:
                continue
            similarities = query_points[:, query_point] @ gallery_points[:, gallery_point].T
            order = np.argsort(-similarities, axis=1, kind="stable")
            target_positions = np.arange(len(object_ids))
            target_ranks = np.argmax(order == target_positions[:, None], axis=1) + 1
            top_predictions = order[:, 0]
            for query_index, object_id in enumerate(object_ids):
                predictions.append(
                    {
                        "partition": partition,
                        "direction": f"point{query_point}_to_point{gallery_point}",
                        "query_modality": query_modality,
                        "gallery_modality": gallery_modality,
                        "query_object_id": object_id,
                        "target_gallery_object_id": object_ids[query_index],
                        "predicted_gallery_object_id": object_ids[
                            int(top_predictions[query_index])
                        ],
                        "target_rank": int(target_ranks[query_index]),
                    }
                )
            ranks.extend(target_ranks.tolist())

    ranks_array = np.asarray(ranks, dtype=np.int64)
    n_objects = len(object_ids)
    return {
        "partition": partition,
        "query_modality": query_modality,
        "gallery_modality": gallery_modality,
        "gallery_size": n_objects,
        "query_count": len(ranks_array),
        "cross_point_only": True,
        "capture_points_per_object": point_count,
        "ordered_cross_point_pairs_per_object": point_count * (point_count - 1),
        "accuracy_top1": float(np.mean(ranks_array == 1)),
        "accuracy_top5": float(np.mean(ranks_array <= min(5, n_objects))),
        "mean_reciprocal_rank": float(np.mean(1.0 / ranks_array)),
        "median_positive_rank": float(np.median(ranks_array)),
        "chance_top1": 1.0 / n_objects,
        "chance_top5": min(5, n_objects) / n_objects,
        "predictions": predictions,
    }


def rank_cross_points(features: np.ndarray, records: list[dict[str, Any]], partition: str):
    return rank_cross_modal(
        features,
        features,
        records,
        partition,
        query_modality="same",
        gallery_modality="same",
    )


def evaluate(features_dir: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite retrieval report: {output_dir}")
    feature_report_path = features_dir / "feature-smoke.json"
    feature_report = json.loads(feature_report_path.read_text(encoding="utf-8"))
    if feature_report.get("status") != "PASS_FROZEN_FEATURE_RUNTIME_SMOKE_ONLY":
        raise ValueError("feature receipt did not pass runtime integrity checks")
    if feature_report.get("evaluation_partition_accessed") is not False:
        raise ValueError("feature receipt records evaluation partition access")
    records = feature_report["records"]
    results = {}
    array_hashes = {}
    features_by_modality = {}
    for modality in MODALITIES:
        feature_path = features_dir / f"{modality}-features.npy"
        expected_hash = feature_report["feature_array_sha256"][feature_path.name]
        if sha256_file(feature_path) != expected_hash:
            raise ValueError(f"feature array hash mismatch: {feature_path}")
        features = np.load(feature_path, allow_pickle=False)
        if list(features.shape) != feature_report["feature_shapes"][modality]:
            raise ValueError(f"feature array shape mismatch: {modality}")
        array_hashes[feature_path.name] = expected_hash
        features_by_modality[modality] = features
        results[modality] = {
            partition: rank_cross_points(features, records, partition)
            for partition in ("train", "validation")
        }

    direct_cross_modal = {}
    if features_by_modality["image"].shape[1] == features_by_modality["tactile"].shape[1]:
        for query_name, gallery_name in (("image", "tactile"), ("tactile", "image")):
            direct_cross_modal[f"{query_name}_to_{gallery_name}"] = {
                partition: rank_cross_modal(
                    features_by_modality[query_name],
                    features_by_modality[gallery_name],
                    records,
                    partition,
                    query_modality=query_name,
                    gallery_modality=gallery_name,
                )
                for partition in ("train", "validation")
            }

    summary = {
        "status": "PASS_FROZEN_CROSS_POINT_RETRIEVAL_DIAGNOSTIC",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "feature_receipt_sha256": sha256_file(feature_report_path),
        "feature_array_sha256": array_hashes,
        "dataset_revision": feature_report["dataset_revision"],
        "train_validation_only": True,
        "evaluation_partition_accessed": False,
        "training": False,
        "task": (
            "same- and direct-compatible cross-modality object retrieval "
            "across different points"
        ),
        "probability_metrics_reported": False,
        "within_modality": results,
        "zero_shot_cross_modal": direct_cross_modal,
        "interpretation_limit": (
            "This measures whether each frozen representation distinguishes the same object "
            "across two points in this dataset. It is not a cross-modal decision result or a "
            "semantic object-recognition benchmark."
        ),
    }
    output_dir.mkdir(parents=True)
    output_path = output_dir / "retrieval-diagnostic.json"
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    summary["report_sha256"] = sha256_file(output_path)
    (output_dir / "retrieval-diagnostic.sha256").write_text(
        summary["report_sha256"] + "  retrieval-diagnostic.json\n", encoding="ascii"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.features_dir, args.output_dir)
    compact = {
        section: {
            method: {
                partition: {
                    key: value
                    for key, value in metrics.items()
                    if key not in {"predictions", "partition"}
                }
                for partition, metrics in partitions.items()
            }
            for method, partitions in methods.items()
        }
        for section, methods in (
            ("within_modality", result["within_modality"]),
            ("zero_shot_cross_modal", result["zero_shot_cross_modal"]),
        )
    }
    print(json.dumps({"status": result["status"], "results": compact}, indent=2))


if __name__ == "__main__":
    main()
