"""Run a fixed train-only initial-frame appearance baseline for Physion++."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scripts.audit_physionpp_factor_only_baseline import (
        DEFAULT_TRAIN_SHA256,
        _binary_metrics,
        group_fold,
    )
    from scripts.audit_physionpp_train_data_quality import (
        _relative_media_path,
        audit_train_rows,
        load_train_rows,
        sha256_file,
    )
except ModuleNotFoundError:
    from audit_physionpp_factor_only_baseline import (
        DEFAULT_TRAIN_SHA256,
        _binary_metrics,
        group_fold,
    )
    from audit_physionpp_train_data_quality import (
        _relative_media_path,
        audit_train_rows,
        load_train_rows,
        sha256_file,
    )

FOLD_COUNT = 5
GRID_SIZE = 8
VARIANCE_FLOOR = 0.05


def spatial_rgb_features(frame: np.ndarray, grid_size: int = GRID_SIZE) -> np.ndarray:
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("expected an HxWx3 RGB frame")
    height, width, _channels = frame.shape
    if height % grid_size or width % grid_size:
        raise ValueError("frame dimensions must be divisible by grid_size")
    values = frame.astype(np.float32) / 255.0
    return values.reshape(
        grid_size,
        height // grid_size,
        grid_size,
        width // grid_size,
        3,
    ).mean(axis=(1, 3)).reshape(-1)


def extract_first_frame(path: Path) -> np.ndarray:
    try:
        import av
    except ImportError as error:
        raise RuntimeError("initial-frame extraction requires PyAV") from error
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        frame = next(container.decode(stream), None)
        if frame is None:
            raise ValueError(f"video has no decodable frames: {path}")
        return frame.to_ndarray(format="rgb24")


def gaussian_nb_probabilities(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    *,
    variance_floor: float = VARIANCE_FLOOR,
) -> np.ndarray:
    if train_features.ndim != 2 or test_features.ndim != 2:
        raise ValueError("features must be two-dimensional")
    if train_features.shape[1] != test_features.shape[1]:
        raise ValueError("train/test feature width mismatch")
    if len(train_features) != len(train_targets) or len(train_targets) == 0:
        raise ValueError("training inputs must be non-empty and aligned")
    if set(int(value) for value in train_targets) != {0, 1}:
        raise ValueError("training fold must contain both classes")
    if variance_floor <= 0:
        raise ValueError("variance_floor must be positive")

    log_scores: list[np.ndarray] = []
    for label in (0, 1):
        class_features = train_features[train_targets == label]
        mean = class_features.mean(axis=0)
        variance = np.maximum(class_features.var(axis=0), variance_floor)
        prior = len(class_features) / len(train_targets)
        log_likelihood = -0.5 * np.sum(
            np.log(2.0 * math.pi * variance)
            + ((test_features - mean) ** 2) / variance,
            axis=1,
        )
        log_scores.append(log_likelihood + math.log(prior))
    scores = np.stack(log_scores, axis=1)
    scores -= scores.max(axis=1, keepdims=True)
    probabilities = np.exp(scores)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities[:, 1]


def run_initial_frame_audit(
    train_jsonl: Path,
    corpus_root: Path,
    output_dir: Path,
    *,
    expected_sha256: str = DEFAULT_TRAIN_SHA256,
) -> dict[str, Any]:
    manifest_bytes = train_jsonl.read_bytes()
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    if manifest_sha != expected_sha256.lower():
        raise ValueError(f"train manifest SHA-256 mismatch: {manifest_sha}")
    rows = load_train_rows(train_jsonl)
    audit = audit_train_rows(rows, corpus_root)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite audit output: {output_dir}")

    feature_rows: list[dict[str, Any]] = []
    media_hashes: list[dict[str, str]] = []
    for row in rows:
        media = row["media"][0]
        relative_path = _relative_media_path(media["path"])
        media_path = corpus_root.joinpath(*relative_path.parts)
        actual_sha = sha256_file(media_path)
        if actual_sha != media["sha256"]:
            raise ValueError(f"media hash mismatch: {row['id']}")
        feature = spatial_rgb_features(extract_first_frame(media_path))
        if not np.isfinite(feature).all():
            raise ValueError(f"non-finite first-frame feature: {row['id']}")
        feature_rows.append(
            {
                "sample_id": row["id"],
                "source_asset_group_id": row["source_asset_group_id"],
                "target": int(row["source_target"]),
                "feature": feature,
            }
        )
        media_hashes.append({"sample_id": row["id"], "sha256": actual_sha})

    groups_by_fold: dict[int, set[str]] = defaultdict(set)
    for row in feature_rows:
        groups_by_fold[group_fold(row["source_asset_group_id"], FOLD_COUNT)].add(
            row["source_asset_group_id"]
        )
    if set(groups_by_fold) != set(range(FOLD_COUNT)):
        raise ValueError("fixed grouped folds are not all populated")

    predictions: list[dict[str, Any]] = []
    folds: list[dict[str, Any]] = []
    all_indices = np.arange(len(feature_rows))
    features = np.stack([row["feature"] for row in feature_rows])
    targets = np.asarray([row["target"] for row in feature_rows], dtype=np.int64)
    fold_assignments = np.asarray(
        [
            group_fold(row["source_asset_group_id"], FOLD_COUNT)
            for row in feature_rows
        ],
        dtype=np.int64,
    )
    for fold in range(FOLD_COUNT):
        test_indices = all_indices[fold_assignments == fold]
        train_indices = all_indices[fold_assignments != fold]
        train_groups = {
            feature_rows[index]["source_asset_group_id"] for index in train_indices
        }
        test_groups = {
            feature_rows[index]["source_asset_group_id"] for index in test_indices
        }
        if train_groups & test_groups:
            raise ValueError(f"source asset group crosses folds: {fold}")
        prior_yes = (targets[train_indices].sum() + 1.0) / (len(train_indices) + 2.0)
        probabilities = gaussian_nb_probabilities(
            features[train_indices], targets[train_indices], features[test_indices]
        )
        fold_prior = np.full(len(test_indices), prior_yes, dtype=np.float64)
        folds.append(
            {
                "fold": fold,
                "train_examples": len(train_indices),
                "test_examples": len(test_indices),
                "train_groups": len(train_groups),
                "test_groups": len(test_groups),
                "prior_baseline": _binary_metrics(
                    fold_prior.tolist(), targets[test_indices].tolist()
                ),
                "initial_frame_only": _binary_metrics(
                    probabilities.tolist(), targets[test_indices].tolist()
                ),
            }
        )
        for index, probability in zip(test_indices, probabilities, strict=True):
            row = feature_rows[int(index)]
            predictions.append(
                {
                    "sample_id": row["sample_id"],
                    "source_asset_group_id": row["source_asset_group_id"],
                    "fold": fold,
                    "target": "Yes" if row["target"] else "No",
                    "probability_yes": float(probability),
                    "prediction": "Yes" if probability >= 0.5 else "No",
                }
            )

    predictions.sort(key=lambda item: item["sample_id"])
    output_dir.mkdir(parents=True)
    predictions_path = output_dir / "oof-predictions.jsonl"
    predictions_path.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in predictions),
        encoding="utf-8",
    )
    media_hash_path = output_dir / "decoded-media-hashes.jsonl"
    media_hash_path.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in media_hashes),
        encoding="utf-8",
    )
    pooled_targets = [int(item["target"] == "Yes") for item in predictions]
    pooled_probabilities = [item["probability_yes"] for item in predictions]
    report = {
        "status": "pass",
        "purpose": (
            "train-only fixed initial-frame appearance diagnostic; "
            "not a video model-quality score or proof of leakage"
        ),
        "train_manifest_sha256": manifest_sha,
        "verified_train_rows_and_media": audit["train_rows"],
        "feature": {
            "definition": "mean RGB per cell on an 8x8 uniform grid of first decoded frame",
            "shape": [GRID_SIZE, GRID_SIZE, 3],
            "normalization": "RGB uint8 converted to float32 in [0,1]",
        },
        "classifier": {
            "name": "Gaussian naive Bayes",
            "variance_floor": VARIANCE_FLOOR,
            "hyperparameter_search": False,
        },
        "fold_policy": "five folds assigned by int(SHA256(source_asset_group_id)[:4])/modulo 5",
        "folds": folds,
        "oof_initial_frame_only": _binary_metrics(
            pooled_probabilities, pooled_targets
        ),
        "oof_predictions_sha256": sha256_file(predictions_path),
        "decoded_media_hashes_sha256": sha256_file(media_hash_path),
        "validation_or_sealed_data_opened": False,
        "gpu_used": False,
        "checkpoint_selected": False,
    }
    (output_dir / "audit-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-train-sha256", default=DEFAULT_TRAIN_SHA256)
    args = parser.parse_args()
    print(
        json.dumps(
            run_initial_frame_audit(
                args.train_jsonl,
                args.corpus_root,
                args.output_dir,
                expected_sha256=args.expected_train_sha256,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
