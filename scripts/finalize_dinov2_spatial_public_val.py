"""Verify and summarize saved predictions from a completed fixed evaluation."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))
sys.path.insert(0, str(REPOSITORY / "src"))

from scripts.evaluate_dinov2_clevr4_spatial_public_val import (  # noqa: E402
    EXPECTED_VAL_IMAGES,
    bootstrap_differences,
    load_jsonl,
    sha256_file,
    sha256_ordered_ids,
)
from tiny_omni_decision.clevr4_data import require_disjoint_media  # noqa: E402
from tiny_omni_decision.clevr4_readout import classification_quality  # noqa: E402
from tiny_omni_decision.dataset import CLEVR4_TAXONOMIES  # noqa: E402


def summarize_predictions(
    labels: list[dict], rows: list[dict], *, expected_images: int = EXPECTED_VAL_IMAGES
) -> tuple[dict, dict]:
    ids = [row["id"] for row in labels]
    if len(ids) != expected_images or len(set(ids)) != expected_images:
        raise ValueError("expected the complete ordered official public-val cohort")
    expected_count = len(ids) * len(CLEVR4_TAXONOMIES)
    if len(rows) != expected_count:
        raise ValueError(f"expected {expected_count} prediction rows, found {len(rows)}")
    labels_by_id = {row["id"]: row for row in labels}
    probabilities = {arm: {task: [] for task in CLEVR4_TAXONOMIES} for arm in ("global", "spatial")}
    targets = {task: [] for task in CLEVR4_TAXONOMIES}
    metric_differences = {
        metric: np.zeros(len(ids), dtype=np.float64) for metric in ("accuracy", "nll", "brier")
    }
    index_by_id = {image_id: index for index, image_id in enumerate(ids)}
    seen: set[str] = set()
    for row_index, row in enumerate(rows):
        image_id = row.get("image_id")
        task = row.get("taxonomy")
        if task not in CLEVR4_TAXONOMIES or image_id not in labels_by_id:
            raise ValueError("prediction row has an unknown image or taxonomy")
        expected_row_index = index_by_id[image_id] * len(CLEVR4_TAXONOMIES)
        expected_row_index += list(CLEVR4_TAXONOMIES).index(task)
        if row_index != expected_row_index or row.get("id") != f"{image_id}:{task}":
            raise ValueError("prediction rows are not in the fixed image/taxonomy order")
        if row["options"] != CLEVR4_TAXONOMIES[task]:
            raise ValueError("prediction candidate order changed")
        target_label = str(labels_by_id[image_id][task])
        if row.get("target") != target_label:
            raise ValueError("saved prediction target does not match the fixed label manifest")
        target = CLEVR4_TAXONOMIES[task].index(target_label)
        targets[task].append(target)
        seen.add(row["id"])
        paired = {}
        for arm in ("global", "spatial"):
            values = torch.tensor(row[f"{arm}_probabilities"], dtype=torch.float64)
            if values.shape != (len(CLEVR4_TAXONOMIES[task]),):
                raise ValueError("prediction vector has an unexpected option count")
            if not torch.isfinite(values).all() or (values < 0).any():
                raise ValueError("prediction vector contains invalid probabilities")
            if not torch.isclose(values.sum(), torch.tensor(1.0, dtype=values.dtype), atol=1e-5):
                raise ValueError("prediction vector is not normalized")
            probabilities[arm][task].append(values.float())
            correct = int(values.argmax().item() == target)
            nll = -math.log(max(float(values[target]), 1e-12))
            one_hot = torch.nn.functional.one_hot(
                torch.tensor(target), len(CLEVR4_TAXONOMIES[task])
            ).to(values.dtype)
            brier = float((values - one_hot).square().sum())
            paired[arm] = (correct, nll, brier)
        image_index = index_by_id[image_id]
        for column, metric in enumerate(("accuracy", "nll", "brier")):
            metric_differences[metric][image_index] += (
                paired["spatial"][column] - paired["global"][column]
            ) / len(CLEVR4_TAXONOMIES)
    if len(seen) != expected_count:
        raise ValueError("saved predictions contain duplicate or missing decisions")

    metrics: dict[str, dict] = {}
    for arm in ("global", "spatial"):
        by_task = {
            task: classification_quality(
                torch.stack(probabilities[arm][task]), torch.tensor(targets[task])
            )
            for task in CLEVR4_TAXONOMIES
        }
        macro = {
            name: float(np.mean([result[name] for result in by_task.values()]))
            for name in ("accuracy", "nll", "brier", "ece", "mean_confidence")
        }
        macro["images_per_taxonomy"] = len(ids)
        macro["decisions"] = expected_count
        macro["by_taxonomy"] = by_task
        metrics[arm] = macro
    return metrics, metric_differences


def finalize(args: argparse.Namespace) -> dict:
    config_path = args.output / "run-config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("status") != "RUNNING" or not config.get("evaluation_only"):
        raise ValueError("output is not an unfinished evaluation-only run")
    if config.get("training_performed") or config.get("checkpoint_selection"):
        raise ValueError("refusing to finalize a run that trained or selected checkpoints")
    failure = json.loads((args.output / "failure.json").read_text(encoding="utf-8"))
    if failure.get("error") != "validation record needs a SHA-256 media identity":
        raise ValueError("saved failure does not match the known post-inference manifest error")

    labels = load_jsonl(args.output / "public-val-labels.jsonl")
    predictions = load_jsonl(args.output / "predictions.jsonl")
    media_rows = load_jsonl(args.output / "public-val-media-manifest.jsonl")
    ids = [row["id"] for row in labels]
    if sha256_ordered_ids(ids) != config["validation_id_order_sha256"]:
        raise ValueError("saved label order does not match the run config")
    if (
        sha256_file(args.output / "public-val-labels.jsonl")
        != config["validation_label_manifest_sha256"]
    ):
        raise ValueError("saved label manifest hash does not match the run config")
    previous_train = load_jsonl(args.media_hashed_dir / "train-media-manifest.jsonl")
    previous_train += load_jsonl(args.media_hashed_dir / "validation-media-manifest.jsonl")
    historical = load_jsonl(args.historical_validation_manifest)
    historical_ids = {row["id"] for row in historical}
    historical_hashes = {row["sha256"] for row in historical}
    prior_public = load_jsonl(args.prior_public_val_dir / "public-val-media-manifest.jsonl")
    prior_by_id = {row["id"]: row["media_sha256"] for row in prior_public}
    if set(prior_by_id) != set(ids) or len(prior_public) != len(ids):
        raise ValueError("prior public-val media manifest does not match the fixed IDs")
    if len(media_rows) != len(ids) or {row["id"] for row in media_rows} != set(ids):
        raise ValueError("saved media manifest is incomplete or contains duplicate IDs")
    normalized_media = []
    for row in media_rows:
        digest = row.get("bytes_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("saved media manifest has an invalid byte hash")
        if digest != prior_by_id[row["id"]] or digest in historical_hashes:
            raise ValueError(
                "saved media bytes differ from the fixed prior or overlap historical val"
            )
        normalized_media.append(
            {"id": row["id"], "asset_id": row["asset_id"], "media_sha256": digest}
        )
    if set(ids) & historical_ids:
        raise ValueError("official public-val IDs overlap historical validation")
    require_disjoint_media(previous_train, normalized_media)
    metrics, differences = summarize_predictions(labels, predictions)

    if not (args.output / "run-config-initial.json").exists():
        shutil.copyfile(config_path, args.output / "run-config-initial.json")
    finished = datetime.now(UTC).isoformat()
    config.update(
        {
            "status": "COMPLETE",
            "finished_utc": finished,
            "recovered_after_post_inference_manifest_schema_error": True,
            "prediction_validation": (
                "all fixed image/taxonomy IDs, targets, candidate orders, "
                "and probabilities verified on CPU"
            ),
            "option_embedding_sha256": None,
            "option_embedding_hash_limitation": (
                "not persisted by the first runner attempt before its "
                "post-inference manifest validation failure"
            ),
        }
    )
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    config_hash = sha256_file(config_path)
    result = {
        **config,
        "run_config_sha256": config_hash,
        "metrics": metrics,
        "spatial_minus_global_macro": {
            name: metrics["spatial"][name] - metrics["global"][name]
            for name in ("accuracy", "nll", "brier", "ece", "mean_confidence")
        },
        "paired_image_bootstrap": bootstrap_differences(differences),
        "finalized_utc": finished,
        "original_post_inference_failure": failure,
        "sealed_audit": False,
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    finalization = {
        "status": "COMPLETE",
        "method": (
            "CPU verification and aggregation of the original complete prediction file; "
            "no model inference rerun"
        ),
        "prediction_rows": len(predictions),
        "run_config_sha256": config_hash,
        "finished_utc": finished,
    }
    (args.output / "finalization.json").write_text(json.dumps(finalization, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--media-hashed-dir", type=Path, required=True)
    parser.add_argument("--historical-validation-manifest", type=Path, required=True)
    parser.add_argument("--prior-public-val-dir", type=Path, required=True)
    args = parser.parse_args()
    result = finalize(args)
    print(
        json.dumps(
            {
                "status": result["status"],
                "metrics": result["metrics"],
                "delta": result["spatial_minus_global_macro"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
