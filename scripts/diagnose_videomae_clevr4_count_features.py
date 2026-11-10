"""Run a fixed count-only linear separability diagnostic on cached train features."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from scripts.audit_clevr4_content_overlap import _load_jsonl, sha256_file

TRAIN_COUNT = 72_010
SELECTOR_COUNT = 1_000
FEATURE_WIDTH = 384
FEATURE_CACHE_SHA256 = "7b6cef6b6451eb7cafd8d918e34fe527fac20fb658ae79c692cac28c9130e4b1"
INPUT_RECEIPT_SHA256 = "350e7d4dec9f113d941f1415dc3bda4e556315ea57baffb89fe2efa62717a16b"
SELECTOR_PREDICTIONS_SHA256 = "603e370f101d2c2f487757245e0a674b4ba1265d7cc995218c96d880d79a2d42"
RIDGE_ALPHA = 0.01
N_FOLDS = 5
SEED = 17


def _hash_id(sample_id: str, *, prefix: str = "seed17:clevr4-count-feature-oof-v1:") -> str:
    return hashlib.sha256(f"{prefix}{sample_id}".encode()).hexdigest()


def stratified_hash_folds(
    ids: list[str], targets: np.ndarray, n_folds: int = N_FOLDS
) -> np.ndarray:
    if len(ids) != len(set(ids)) or targets.shape != (len(ids),):
        raise ValueError("fold inputs must contain one target for every unique image ID")
    if n_folds < 2 or len(ids) < n_folds:
        raise ValueError("at least two folds and one sample per fold are required")
    fold_ids = np.full(len(ids), -1, dtype=np.int8)
    for target in sorted(np.unique(targets).tolist()):
        members = [index for index, value in enumerate(targets) if value == target]
        members.sort(key=lambda index: _hash_id(ids[index]))
        if len(members) < n_folds:
            raise ValueError(f"class {target} has fewer members than folds")
        for rank, index in enumerate(members):
            fold_ids[index] = rank % n_folds
    if np.any(fold_ids < 0):
        raise ValueError("some samples were not assigned to a fold")
    return fold_ids


def ridge_scores(
    fit_features: np.ndarray,
    fit_targets: np.ndarray,
    query_features: np.ndarray,
    *,
    alpha: float = RIDGE_ALPHA,
    n_classes: int = 10,
) -> np.ndarray:
    if fit_features.ndim != 2 or query_features.ndim != 2:
        raise ValueError("features must be two-dimensional")
    if fit_features.shape[1] != query_features.shape[1] or fit_targets.shape != (
        len(fit_features),
    ):
        raise ValueError("feature and target shapes do not agree")
    if (
        len(fit_features) == 0
        or not np.isfinite(fit_features).all()
        or not np.isfinite(query_features).all()
    ):
        raise ValueError("features must be non-empty and finite")
    if np.any(fit_targets < 0) or np.any(fit_targets >= n_classes):
        raise ValueError("target index is outside the configured class set")
    if alpha <= 0:
        raise ValueError("ridge alpha must be positive")

    x_fit = np.asarray(fit_features, dtype=np.float64)
    x_query = np.asarray(query_features, dtype=np.float64)
    mean = x_fit.mean(axis=0)
    scale = x_fit.std(axis=0)
    scale[scale < 1e-8] = 1.0
    x_fit = (x_fit - mean) / scale
    x_query = (x_query - mean) / scale

    target_mean = np.full(n_classes, 0.0, dtype=np.float64)
    target_mean += np.bincount(fit_targets, minlength=n_classes) / len(fit_targets)
    y_fit = np.eye(n_classes, dtype=np.float64)[fit_targets] - target_mean
    gram = (x_fit.T @ x_fit) / len(x_fit)
    gram.flat[:: len(gram) + 1] += alpha
    rhs = (x_fit.T @ y_fit) / len(x_fit)
    weights = np.linalg.solve(gram, rhs)
    return x_query @ weights + target_mean


def _read_feature_prefix(path: Path, rows: int) -> np.ndarray:
    """Read only the leading train/selector rows; never parse the report suffix."""
    with zipfile.ZipFile(path) as archive:
        with archive.open("features.npy") as stream:
            shape, fortran_order, dtype = _read_npy_header(stream)
            if shape != (78_306, FEATURE_WIDTH) or fortran_order or dtype != np.dtype("float32"):
                raise ValueError(
                    f"unexpected feature member header: {shape}, {fortran_order}, {dtype}"
                )
            if rows > shape[0]:
                raise ValueError("requested prefix exceeds frozen cache")
            needed = rows * FEATURE_WIDTH * np.dtype("float32").itemsize
            blocks = bytearray()
            while len(blocks) < needed:
                part = stream.read(min(8 * 1024 * 1024, needed - len(blocks)))
                if not part:
                    raise ValueError(
                        "feature cache ended before the requested train/selector prefix"
                    )
                blocks.extend(part)
            return np.frombuffer(blocks, dtype=np.float32).reshape(rows, FEATURE_WIDTH).copy()


def _read_npy_header(stream: Any) -> tuple[tuple[int, ...], bool, np.dtype[Any]]:
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(stream)
    elif version == (2, 0):
        shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(stream)
    else:
        raise ValueError(f"unsupported NumPy feature member version: {version}")
    return shape, fortran_order, dtype


def _class_metrics(
    targets: np.ndarray, predictions: np.ndarray, class_count: int = 10
) -> dict[str, Any]:
    confusion = np.zeros((class_count, class_count), dtype=np.int64)
    np.add.at(confusion, (targets, predictions), 1)
    supports = confusion.sum(axis=1)
    recalls = np.divide(
        confusion.diagonal(),
        supports,
        out=np.zeros(class_count, dtype=np.float64),
        where=supports > 0,
    )
    return {
        "n": int(len(targets)),
        "accuracy": float(np.mean(targets == predictions)),
        "macro_recall": float(recalls.mean()),
        "class_support": {str(index + 1): int(value) for index, value in enumerate(supports)},
        "class_recall": {str(index + 1): float(value) for index, value in enumerate(recalls)},
        "confusion_counts_rows_target_cols_prediction": confusion.tolist(),
    }


def paired_accuracy_bootstrap(
    targets: np.ndarray,
    first_predictions: np.ndarray,
    second_predictions: np.ndarray,
    *,
    iterations: int = 10_000,
    seed: int = SEED,
) -> dict[str, float]:
    if not (targets.shape == first_predictions.shape == second_predictions.shape):
        raise ValueError("paired bootstrap arrays must be aligned")
    if len(targets) == 0:
        raise ValueError("paired bootstrap requires observations")
    first_correct = (first_predictions == targets).astype(np.float64)
    second_correct = (second_predictions == targets).astype(np.float64)
    differences = second_correct - first_correct
    rng = np.random.default_rng(seed)
    means = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        sampled = rng.integers(0, len(targets), size=len(targets))
        means[index] = differences[sampled].mean()
    return {
        "point_difference_second_minus_first": float(differences.mean()),
        "lower_95_percentile": float(np.quantile(means, 0.025)),
        "upper_95_percentile": float(np.quantile(means, 0.975)),
        "iterations": iterations,
        "seed": seed,
    }


def run(*, inputs: Path, feature_cache: Path, fit_dir: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    started = time.perf_counter()
    receipt = json.loads((inputs / "receipt.json").read_text(encoding="utf-8"))
    if sha256_file(inputs / "receipt.json") != INPUT_RECEIPT_SHA256:
        raise ValueError("frozen input receipt hash differs from the registered run")
    if receipt.get("status") != "FROZEN_VIDEOMAE_CLEVR4_PATH_A_INPUTS":
        raise ValueError("input receipt is not the frozen Path A receipt")
    if any(
        receipt.get(key) != 0
        for key in (
            "cross_split_asset_id_overlap",
            "cross_split_id_overlap",
            "cross_split_media_sha256_overlap",
        )
    ):
        raise ValueError("frozen input receipt reports cross-split overlap")
    config_path = inputs / "config.json"
    if sha256_file(config_path) != receipt.get("config_sha256"):
        raise ValueError("frozen input config hash differs from its receipt")
    expected = {
        "train-labels.jsonl": receipt["file_hashes"]["train-labels.jsonl"],
        "train-media-manifest.jsonl": receipt["file_hashes"]["train-media-manifest.jsonl"],
        "selector-labels.jsonl": receipt["file_hashes"]["selector-labels.jsonl"],
        "selector-media-manifest.jsonl": receipt["file_hashes"]["selector-media-manifest.jsonl"],
        "features-manifest.jsonl": receipt["features_manifest_sha256"],
        "config.json": receipt["config_sha256"],
    }
    for name, digest in expected.items():
        if sha256_file(inputs / name) != digest:
            raise ValueError(f"frozen input hash mismatch: {name}")
    feature_hash = sha256_file(feature_cache)
    if feature_hash != FEATURE_CACHE_SHA256:
        raise ValueError("frozen feature cache hash differs from the registered run")
    predictions_path = fit_dir / "selector-best-predictions.jsonl"
    if sha256_file(predictions_path) != SELECTOR_PREDICTIONS_SHA256:
        raise ValueError("saved selector prediction hash differs from the registered run")

    train_rows = _load_jsonl(inputs / "train-labels.jsonl")
    selector_rows = _load_jsonl(inputs / "selector-labels.jsonl")
    train_media = _load_jsonl(inputs / "train-media-manifest.jsonl")
    selector_media = _load_jsonl(inputs / "selector-media-manifest.jsonl")
    feature_rows = _load_jsonl(inputs / "features-manifest.jsonl")
    if (len(train_rows), len(selector_rows), len(feature_rows)) != (
        TRAIN_COUNT,
        SELECTOR_COUNT,
        78_306,
    ):
        raise ValueError("frozen input counts differ from preregistered counts")
    train_ids = [str(row["id"]) for row in train_rows]
    selector_ids = [str(row["id"]) for row in selector_rows]
    if len(set(train_ids)) != TRAIN_COUNT or len(set(selector_ids)) != SELECTOR_COUNT:
        raise ValueError("duplicate train or selector image IDs")
    if set(train_ids).intersection(selector_ids):
        raise ValueError("train and selector IDs overlap")
    media_hashes = [str(row["media_sha256"]) for row in train_media + selector_media]
    if len(media_hashes) != len(set(media_hashes)):
        raise ValueError("train and selector media identities overlap")
    prefix_rows = feature_rows[: TRAIN_COUNT + SELECTOR_COUNT]
    prefix_ids = [str(row["id"]) for row in prefix_rows]
    if prefix_ids != train_ids + selector_ids:
        raise ValueError("feature-cache prefix is not ordered train then selector")
    for label_rows, media_rows in ((train_rows, train_media), (selector_rows, selector_media)):
        if [str(row["id"]) for row in label_rows] != [str(row["id"]) for row in media_rows]:
            raise ValueError("label and media-manifest order differs")
        if any(
            str(label["asset_id"]) != str(media["asset_id"])
            for label, media in zip(label_rows, media_rows, strict=True)
        ):
            raise ValueError("label and media asset IDs differ")
    for index, row in enumerate(prefix_rows):
        media_row = (train_media + selector_media)[index]
        if row.get("media_sha256") != media_row.get("media_sha256"):
            raise ValueError("feature media hash differs from the frozen input manifest")

    # The compressed feature cache contains train, selector and reporting rows.
    # Only the train+selector prefix is decompressed into the analysis array.
    prefix_features = _read_feature_prefix(feature_cache, TRAIN_COUNT + SELECTOR_COUNT)
    if not np.isfinite(prefix_features).all():
        raise ValueError("train/selector feature prefix contains NaN or Inf")
    train_x = prefix_features[:TRAIN_COUNT]
    selector_x = prefix_features[TRAIN_COUNT:]
    train_y = np.asarray([int(row["count"]) - 1 for row in train_rows], dtype=np.int64)
    selector_y = np.asarray([int(row["count"]) - 1 for row in selector_rows], dtype=np.int64)
    if np.any((train_y < 0) | (train_y >= 10)) or np.any((selector_y < 0) | (selector_y >= 10)):
        raise ValueError("count labels are outside 1..10")

    folds = stratified_hash_folds(train_ids, train_y)
    oof_predictions = np.full(TRAIN_COUNT, -1, dtype=np.int64)
    fold_metrics = []
    for fold in range(N_FOLDS):
        query = np.flatnonzero(folds == fold)
        fit = np.flatnonzero(folds != fold)
        scores = ridge_scores(train_x[fit], train_y[fit], train_x[query])
        oof_predictions[query] = scores.argmax(axis=1)
        fold_metrics.append(_class_metrics(train_y[query], oof_predictions[query]))
    if np.any(oof_predictions < 0):
        raise ValueError("some train-only out-of-fold examples lack predictions")

    selector_scores = ridge_scores(train_x, train_y, selector_x)
    ridge_selector_predictions = selector_scores.argmax(axis=1)
    saved_predictions = _load_jsonl(predictions_path)
    count_prediction_by_id = {
        str(row["image_id"]): int(row["prediction"])
        for row in saved_predictions
        if row.get("task") == "count"
    }
    if len(count_prediction_by_id) != SELECTOR_COUNT or set(count_prediction_by_id) != set(
        selector_ids
    ):
        raise ValueError("saved MLP selector count predictions do not align with selector labels")
    mlp_selector_predictions = np.asarray(
        [count_prediction_by_id[sample_id] - 1 for sample_id in selector_ids]
    )

    report = {
        "schema_version": 1,
        "experiment_id": "videomae-clevr4-count-feature-diagnostic-20261110",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_commit": _git_commit(),
        "diagnostic_scope": (
            "linear separability of the frozen 384-D global image feature "
            "for one synthetic count taxonomy task"
        ),
        "access_boundary": {
            "train_labels_read": True,
            "selector_labels_read": True,
            "report_labels_read": False,
            "report_predictions_read": False,
            "report_feature_rows_decompressed_into_analysis_array": False,
            "official_validation_read": False,
            "sealed_audit_read": False,
            "feature_rows_used": TRAIN_COUNT + SELECTOR_COUNT,
        },
        "input_hashes": {
            "input_receipt_sha256": INPUT_RECEIPT_SHA256,
            "feature_cache_sha256": feature_hash,
            "features_manifest_sha256": expected["features-manifest.jsonl"],
            "train_labels_sha256": expected["train-labels.jsonl"],
            "selector_labels_sha256": expected["selector-labels.jsonl"],
            "selector_mlp_predictions_sha256": SELECTOR_PREDICTIONS_SHA256,
            "train_id_order_sha256": receipt["train_id_order_sha256"],
            "selector_id_order_sha256": receipt["selector_id_order_sha256"],
        },
        "recipe": {
            "readout": "multiclass least-squares ridge over one-hot count classes",
            "folds": N_FOLDS,
            "fold_assignment": (
                "within-class SHA256(seed17:clevr4-count-feature-oof-v1:<image_id>) order modulo 5"
            ),
            "standardization": "mean/std fit on each fold's fit rows only; std below 1e-8 set to 1",
            "ridge_alpha": RIDGE_ALPHA,
            "accuracy_selection_or_tuning": False,
            "probabilities_or_calibration": False,
        },
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "train_only_oof": {
            **_class_metrics(train_y, oof_predictions),
            "folds": fold_metrics,
        },
        "fixed_selector_diagnostic": {
            "ridge": _class_metrics(selector_y, ridge_selector_predictions),
            "selected_mlp_reference": _class_metrics(selector_y, mlp_selector_predictions),
            "paired_accuracy_bootstrap_ridge_minus_selected_mlp": paired_accuracy_bootstrap(
                selector_y, mlp_selector_predictions, ridge_selector_predictions
            ),
            "interpretation": "descriptive post-selection comparison; not used to choose a model",
        },
    }
    report["elapsed_seconds"] = time.perf_counter() - started
    report["runner_sha256"] = sha256_file(Path(__file__))
    output.mkdir(parents=True, exist_ok=False)
    (output / "diagnostic.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def _git_commit() -> str:
    import subprocess

    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    )
    return completed.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--fit-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(
        inputs=args.inputs,
        feature_cache=args.feature_cache,
        fit_dir=args.fit_dir,
        output=args.output,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "train_oof": report["train_only_oof"]["accuracy"],
                "selector_ridge": report["fixed_selector_diagnostic"]["ridge"]["accuracy"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
