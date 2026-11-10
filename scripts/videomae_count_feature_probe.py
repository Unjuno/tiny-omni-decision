"""Prepare isolated Clevr-4 count inputs and run one fixed linear probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from scripts.audit_clevr4_content_overlap import _load_jsonl, sha256_file

EXPECTED_COUNTS = {"train": 72_010, "selector": 1_000}
EXPECTED_INPUT_HASHES = {
    "train-media-manifest.jsonl": (
        "e48687af785a74864d2d4eb6e632a50d381d821a3ade8c2227eda7624647f410"
    ),
    "selector-media-manifest.jsonl": (
        "4c551e91d4e02bd896380476d0e59f1348a01ed3f95a5eed7e0d39cb9a311dee"
    ),
    "train-labels.jsonl": "45de9d36d557619789dd6160e808f7ff0c531f06a298f0b2af80c9720103b695",
    "selector-labels.jsonl": "c08af3839089638d081dd31c1118be7256418e6c2440a70a55f79e50456fb920",
}
EXPECTED_CONFIG_SHA256 = "97f32fd4a2172526c67265f7cd74a153c993fd2e05992ce0b4899de8a04f160c"


def _order_hash(ids: list[str]) -> str:
    return hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest()


def build_allowed_rows(
    media_by_split: dict[str, list[dict[str, Any]]],
    labels_by_split: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Join only the explicitly supplied train and selector rows in order."""
    all_media: list[dict[str, Any]] = []
    all_labels: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_assets: set[str] = set()
    seen_hashes: set[str] = set()
    for split in ("train", "selector"):
        media_rows = media_by_split[split]
        label_rows = labels_by_split[split]
        if len(media_rows) != len(label_rows):
            raise ValueError(f"{split} media/label counts differ")
        ids = [str(row["id"]) for row in media_rows]
        label_ids = [str(row["id"]) for row in label_rows]
        if ids != label_ids or len(ids) != len(set(ids)):
            raise ValueError(f"{split} media/label ID order differs or contains duplicates")
        for media, label in zip(media_rows, label_rows, strict=True):
            sample_id = str(media["id"])
            asset_id = str(media["asset_id"])
            media_hash = str(media["media_sha256"])
            count = int(label["count"])
            if count not in range(1, 11):
                raise ValueError(f"unexpected count target for {sample_id}")
            if sample_id in seen_ids or asset_id in seen_assets or media_hash in seen_hashes:
                raise ValueError("train/selector ID, asset, or media hash overlap")
            seen_ids.add(sample_id)
            seen_assets.add(asset_id)
            seen_hashes.add(media_hash)
            all_media.append(
                {
                    "id": sample_id,
                    "asset_id": asset_id,
                    "archive_member": str(media["archive_member"]),
                    "media_sha256": media_hash,
                    "split": split,
                }
            )
            all_labels.append({"id": sample_id, "count": count, "split": split})
    return all_media, all_labels


def prepare(inputs: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    receipt = json.loads((inputs / "receipt.json").read_text(encoding="utf-8"))
    if receipt.get("config_sha256") != EXPECTED_CONFIG_SHA256:
        raise ValueError("frozen source config hash differs")
    loaded: dict[str, list[dict[str, Any]]] = {}
    for name, expected_hash in EXPECTED_INPUT_HASHES.items():
        path = inputs / name
        if sha256_file(path) != expected_hash or receipt["file_hashes"].get(name) != expected_hash:
            raise ValueError(f"frozen input hash mismatch: {name}")
        loaded[name] = _load_jsonl(path)
    media, labels = build_allowed_rows(
        {
            "train": loaded["train-media-manifest.jsonl"],
            "selector": loaded["selector-media-manifest.jsonl"],
        },
        {"train": loaded["train-labels.jsonl"], "selector": loaded["selector-labels.jsonl"]},
    )
    counts = {split: sum(row["split"] == split for row in media) for split in EXPECTED_COUNTS}
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"unexpected fixed train/selector counts: {counts}")
    if (
        _order_hash([row["id"] for row in media[: counts["train"]]])
        != receipt["train_id_order_sha256"]
    ):
        raise ValueError("train ID order differs from frozen receipt")
    if (
        _order_hash([row["id"] for row in media[counts["train"] :]])
        != receipt["selector_id_order_sha256"]
    ):
        raise ValueError("selector ID order differs from frozen receipt")

    output.mkdir(parents=True, exist_ok=False)
    media_path = output / "media-manifest.jsonl"
    labels_path = output / "labels.jsonl"
    media_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in media), encoding="utf-8"
    )
    labels_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in labels), encoding="utf-8"
    )
    report = {
        "status": "FROZEN_TRAIN_SELECTOR_ONLY_COUNT_PROBE_INPUTS",
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "source_input_hashes": EXPECTED_INPUT_HASHES,
        "source_input_ids": {split: EXPECTED_COUNTS[split] for split in EXPECTED_COUNTS},
        "train_id_order_sha256": receipt["train_id_order_sha256"],
        "selector_id_order_sha256": receipt["selector_id_order_sha256"],
        "media_manifest_sha256": sha256_file(media_path),
        "labels_sha256": sha256_file(labels_path),
        "counts": counts,
        "report_pool_read": False,
        "official_validation_read": False,
        "sealed_audit_read": False,
    }
    (output / "input-receipt.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def ridge_predict(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    selector_features: np.ndarray,
    regularization: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a standardized, centered one-hot ridge readout with fixed lambda."""
    if train_features.ndim != 2 or selector_features.ndim != 2:
        raise ValueError("feature arrays must be rank two")
    if train_features.shape[1] != selector_features.shape[1]:
        raise ValueError("feature widths differ")
    if train_features.shape[0] != train_labels.shape[0] or not train_features.shape[0]:
        raise ValueError("training features/labels are empty or misaligned")
    if regularization != 1.0:
        raise ValueError("this preregistered diagnostic fixes ridge lambda at 1.0")
    classes = np.arange(1, 11, dtype=np.int64)
    if not np.array_equal(np.unique(train_labels), classes):
        raise ValueError("training labels must cover the fixed count classes 1..10")
    if not np.isfinite(train_features).all() or not np.isfinite(selector_features).all():
        raise ValueError("features must be finite")

    train = train_features.astype(np.float64, copy=False)
    selector = selector_features.astype(np.float64, copy=False)
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale == 0] = 1.0
    train_z = (train - mean) / scale
    selector_z = (selector - mean) / scale
    target = np.eye(classes.size, dtype=np.float64)[train_labels - 1]
    target_mean = target.mean(axis=0)
    centered_target = target - target_mean
    gram = (train_z.T @ train_z) / train_z.shape[0]
    rhs = (train_z.T @ centered_target) / train_z.shape[0]
    coefficients = np.linalg.solve(gram + np.eye(gram.shape[0]) * regularization, rhs)
    scores = selector_z @ coefficients + target_mean
    return classes[scores.argmax(axis=1)], scores


def _summarize(selector_labels: list[int], predictions: np.ndarray) -> dict[str, Any]:
    target = np.asarray(selector_labels, dtype=np.int64)
    confusion = np.zeros((10, 10), dtype=np.int64)
    for actual, predicted in zip(target, predictions, strict=True):
        confusion[actual - 1, int(predicted) - 1] += 1
    support = confusion.sum(axis=1)
    recall = np.divide(
        confusion.diagonal(), support, out=np.zeros(10, dtype=np.float64), where=support > 0
    )
    return {
        "accuracy": float(np.mean(target == predictions)),
        "macro_recall": float(recall.mean()),
        "per_class": [
            {"count": i + 1, "support": int(support[i]), "recall": float(recall[i])}
            for i in range(10)
        ],
        "confusion_matrix_rows_actual_columns_predicted": confusion.tolist(),
    }


def analyze(prepared: Path, features_dir: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    receipt = json.loads((prepared / "input-receipt.json").read_text(encoding="utf-8"))
    media_path = prepared / "media-manifest.jsonl"
    labels_path = prepared / "labels.jsonl"
    if (
        sha256_file(media_path) != receipt["media_manifest_sha256"]
        or sha256_file(labels_path) != receipt["labels_sha256"]
    ):
        raise ValueError("prepared input hashes differ from receipt")
    feature_result = json.loads((features_dir / "result.json").read_text(encoding="utf-8"))
    feature_path = features_dir / "frozen-features.npz"
    if sha256_file(feature_path) != feature_result["feature_sha256"]:
        raise ValueError("feature cache hash differs from extraction receipt")
    expected_rows = _load_jsonl(media_path)
    labels = _load_jsonl(labels_path)
    if feature_result["input_manifest_sha256"] != sha256_file(media_path):
        raise ValueError("feature cache was extracted from a different manifest")
    with np.load(feature_path, allow_pickle=False) as cache:
        ids = cache["ids"].astype(str).tolist()
        features = cache["features"]
    expected_ids = [str(row["id"]) for row in expected_rows]
    if ids != expected_ids or features.shape != (len(expected_rows), 384):
        raise ValueError("feature IDs/order/shape differ from the allowed input manifest")
    if len(labels) != len(ids) or [str(row["id"]) for row in labels] != ids:
        raise ValueError("labels are not aligned to feature IDs")
    if any(row["split"] not in {"train", "selector"} for row in labels):
        raise ValueError("non-allowed split is present")
    train_mask = np.asarray([row["split"] == "train" for row in labels])
    selector_mask = ~train_mask
    if (
        int(train_mask.sum()) != EXPECTED_COUNTS["train"]
        or int(selector_mask.sum()) != EXPECTED_COUNTS["selector"]
    ):
        raise ValueError("train/selector counts differ from fixed inputs")
    started = time.perf_counter()
    predictions, scores = ridge_predict(
        features[train_mask],
        np.asarray([int(row["count"]) for row in labels if row["split"] == "train"]),
        features[selector_mask],
    )
    elapsed = time.perf_counter() - started
    selector_rows = [row for row in labels if row["split"] == "selector"]
    metrics = _summarize([int(row["count"]) for row in selector_rows], predictions)
    output.mkdir(parents=True, exist_ok=False)
    prediction_path = output / "selector-predictions.jsonl"
    prediction_path.write_text(
        "".join(
            json.dumps(
                {
                    "id": row["id"],
                    "target": int(row["count"]),
                    "prediction": int(pred),
                    "scores": score.tolist(),
                },
                sort_keys=True,
            )
            + "\n"
            for row, pred, score in zip(selector_rows, predictions, scores, strict=True)
        ),
        encoding="utf-8",
    )
    result = {
        "status": "COMPLETE_EXPOSED_SELECTOR_LINEAR_SEPARABILITY_DIAGNOSTIC",
        "method": "train-standardized centered one-hot closed-form ridge; lambda=1.0",
        "train_count": int(train_mask.sum()),
        "selector_count": int(selector_mask.sum()),
        "train_labels_sha256": sha256_file(labels_path),
        "feature_sha256": sha256_file(feature_path),
        "encoder_source_sha256": feature_result["encoder_source_sha256"],
        "encoder_checkpoint_sha256": feature_result["encoder_checkpoint_sha256"],
        "selector_metrics": metrics,
        "fit_predict_seconds": elapsed,
        "prediction_sha256": sha256_file(prediction_path),
        "report_pool_read": False,
        "official_validation_read": False,
        "sealed_audit_read": False,
    }
    (output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--inputs", type=Path, required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    analyze_parser = commands.add_parser("analyze")
    analyze_parser.add_argument("--prepared", type=Path, required=True)
    analyze_parser.add_argument("--features", type=Path, required=True)
    analyze_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.inputs, args.output)
    else:
        result = analyze(args.prepared, args.features, args.output)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
