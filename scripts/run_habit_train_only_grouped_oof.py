"""Measure fixed readouts with grouped OOF using HABIT train rows only."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import Tensor, nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import train_habit_temporal_decision as base

FOLD_COUNT = 5
EPOCHS = 8
CURVE_EPOCHS = {1, 2, 4, 8}


def connected_components(examples: list[dict[str, Any]], indices: list[int]) -> list[list[int]]:
    """Connect rows sharing a task group or video identity."""
    parent: dict[str, str] = {}

    def find(node: str) -> str:
        parent.setdefault(node, node)
        if parent[node] != node:
            parent[node] = find(parent[node])
        return parent[node]

    def union(left: str, right: str) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for index in indices:
        row = examples[index]
        union(f"task:{row['sid']}", f"video:{row['video_sha256']}")

    components: dict[str, list[int]] = defaultdict(list)
    for index in indices:
        components[find(f"task:{examples[index]['sid']}")].append(index)
    return sorted(components.values(), key=lambda rows: (min(rows), len(rows)))


def assign_folds(
    examples: list[dict[str, Any]], indices: list[int], *, seed: int = base.SEED,
    fold_count: int = FOLD_COUNT,
) -> list[list[int]]:
    """Greedily balance connected groups without splitting task/video identity."""
    if fold_count < 2:
        raise ValueError("fold_count must be at least two")
    components = connected_components(examples, indices)
    if len(components) < fold_count:
        raise ValueError("not enough independent task/video components for folds")
    rng = random.Random(seed)
    tie_break = {position: rng.random() for position in range(len(components))}
    ordered = sorted(
        enumerate(components),
        key=lambda item: (-len(item[1]), tie_break[item[0]]),
    )
    folds: list[list[int]] = [[] for _ in range(fold_count)]
    fold_component_counts = [0] * fold_count
    for _, component in ordered:
        destination = min(
            range(fold_count),
            key=lambda fold: (len(folds[fold]), fold_component_counts[fold], fold),
        )
        folds[destination].extend(component)
        fold_component_counts[destination] += 1
    return [sorted(fold) for fold in folds]


@torch.inference_mode()
def predict_logits(
    model: nn.Module, mode: str, indices: list[int], examples: list[dict[str, Any]],
    arrays: dict[str, np.ndarray], context: Tensor, option_rows: list[Tensor],
    sensor_mean: Tensor, sensor_std: Tensor, *, donors: list[int] | None = None,
    donor_stream: str | None = None,
) -> dict[int, Tensor]:
    model.eval()
    rng = random.Random(base.SEED)
    result: dict[int, Tensor] = {}
    for start in range(0, len(indices), base.BATCH_SIZE):
        batch_indices = indices[start:start + base.BATCH_SIZE]
        donor_map = None
        if donors is not None and donor_stream == "video":
            donor_map = donors
        batch = base._batch(
            batch_indices, examples, arrays, context, option_rows, sensor_mean,
            sensor_std, mode, shuffle_options=False, rng=rng,
            video_donors=donor_map,
            sensor_donors=donor_map if donor_stream == "sensor" else None,
        )
        logits, _ = base._forward(model, mode, batch)
        for row_position, source_index in enumerate(batch_indices):
            count = len(examples[source_index]["options"])
            result[source_index] = logits[row_position, :count].detach().cpu().float()
    return result


def evaluate_oof(
    predictions: dict[int, Tensor], train_indices: list[int], examples: list[dict[str, Any]],
) -> dict[str, Any]:
    ordered = sorted(train_indices)
    if set(predictions) != set(ordered):
        raise ValueError("OOF prediction IDs do not cover the train rows exactly once")
    metrics = base._metrics(
        [predictions[index] for index in ordered],
        [int(examples[index]["target_index"]) for index in ordered],
        [str(examples[index]["sid"]) for index in ordered],
    )
    metrics["predictions"] = [
        {
            "sample_id": examples[index]["id"],
            "task_group": examples[index]["sid"],
            "video_sha256": examples[index]["video_sha256"],
            "target_index": int(examples[index]["target_index"]),
            "prediction_index": int(np.argmax(metrics["probabilities"][position])),
            "probabilities": metrics["probabilities"][position],
        }
        for position, index in enumerate(ordered)
    ]
    return metrics


def _fold_train_normalizer(
    arrays: dict[str, np.ndarray], train_indices: list[int],
) -> tuple[Tensor, Tensor]:
    train_rows = torch.from_numpy(arrays["sensor_sequence"][train_indices].astype(np.float32))
    flat = train_rows.reshape(-1, train_rows.shape[-1])
    mean = flat.mean(0)
    std = flat.std(0, unbiased=False).clamp_min(1e-4)
    return mean, std


def validate_oof_config(config: dict[str, Any]) -> str:
    """Reject any config that changes the reviewed train-only OOF protocol."""
    training = config.get("training", {})
    data = config.get("data", {})
    evaluation = config.get("evaluation", {})
    prohibitions = config.get("prohibitions", {})
    expected = (
        config.get("schema_version") == 1
        and config.get("experiment_id") == "habit-train-only-grouped-oof-seed17-data-scale-v1"
        and config.get("seed") == base.SEED
        and config.get("device") == "cpu"
        and training.get("epochs") == EPOCHS
        and training.get("batch_size") == base.BATCH_SIZE
        and training.get("learning_rate") == base.LEARNING_RATE
        and training.get("weight_decay") == base.WEIGHT_DECAY
        and training.get("checkpoint_selection") == "none"
        and training.get("early_stopping") is False
        and data.get("package_revision") == "bae97d251907743f8fb6db7e70c61d361df4dc1e"
        and data.get("included_split") == "train"
        and data.get("train_examples") == 606
        and data.get("task_video_components") == 37
        and data.get("folds") == FOLD_COUNT
        and data.get("validation_split_used") is False
        and evaluation.get("sealed_audit_accessed") is False
        and evaluation.get("development_validation_metrics_computed") is False
        and prohibitions.get("sealed_audit_accessed") is False
        and prohibitions.get("hyperparameter_search") is False
        and prohibitions.get("checkpoint_selection") is False
    )
    if not expected:
        raise ValueError("OOF config differs from the fixed train-only protocol")
    return str(config["experiment_id"])


def run(package_dir: Path, text_model: Path, output: Path, config_path: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    output.mkdir(parents=True)
    torch.manual_seed(base.SEED)
    torch.set_num_threads(4)
    started_at = datetime.now(UTC).isoformat()
    config_bytes = config_path.read_bytes()
    config = yaml.safe_load(config_bytes)
    experiment_id = validate_oof_config(config)
    package_report_path = package_dir / "package-report.json"
    examples_path = package_dir / "transition-examples.json"
    feature_path = package_dir / "habit-transition-features.npz"
    expected_hashes = {
        "package_report_sha256": config["data"]["package_report_sha256"],
        "transition_examples_sha256": config["data"]["transition_examples_sha256"],
        "feature_cache_sha256": config["data"]["feature_cache_sha256"],
    }
    actual_hashes = {
        "package_report_sha256": base.sha256_file(package_report_path),
        "transition_examples_sha256": base.sha256_file(examples_path),
        "feature_cache_sha256": base.sha256_file(feature_path),
    }
    if actual_hashes != expected_hashes:
        raise ValueError("frozen train package hashes differ from the reviewed config")
    package_report = json.loads(package_report_path.read_text(encoding="utf-8"))
    if (package_report.get("status") != "complete_text_aligned_frozen_feature_package"
            or package_report.get("sealed_audit_accessed") is not False
            or package_report.get("action_columns_read") is not False
            or package_report.get("training_performed") is not False
            or base.sha256_file(examples_path)
            != package_report.get("transition_examples_sha256")
            or base.sha256_file(feature_path) != package_report.get("feature_cache_sha256")):
        raise ValueError("frozen train package provenance or hashes do not pass")
    example_package = json.loads(examples_path.read_text(encoding="utf-8"))
    if (example_package.get("source_revision") != "bae97d251907743f8fb6db7e70c61d361df4dc1e"
            or example_package.get("sealed_audit_accessed") is not False
            or example_package.get("action_columns_read") is not False):
        raise ValueError("train package source/scope differs from the fixed protocol")
    all_ids = [row["id"] for row in example_package["records"]]
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("source sample IDs are not unique")
    original_train_positions = [
        position for position, row in enumerate(example_package["records"])
        if row.get("split") == "train"
    ]
    examples = [example_package["records"][position] for position in original_train_positions]
    if (not examples or any(row.get("split") != "train" for row in examples)
            or len(examples) != config["data"].get("train_examples")):
        raise ValueError(f"train rows do not match config, got {len(examples)}")
    with np.load(feature_path, allow_pickle=False) as cached:
        ids = cached["ids"].tolist()
        if ids != all_ids:
            raise ValueError("feature cache identifiers differ from frozen examples")
        arrays = {
            name: cached[name][original_train_positions].copy()
            for name in ("ids", "video_features", "sensor_sequence", "option_count", "target_index")
        }
    row_count = len(examples)
    if (arrays["video_features"].shape != (row_count, 8, 768)
            or arrays["sensor_sequence"].shape != (row_count, 8, 14)):
        raise ValueError("train feature cache has unexpected shapes")
    for index, row in enumerate(examples):
        if (len(row["options"]) < 2 or len(row["options"]) != arrays["option_count"][index]
                or not 0 <= row["target_index"] < len(row["options"])
                or row["target_index"] != arrays["target_index"][index]
                or len(set(row["options"])) != len(row["options"])
                or not row["question"].strip()):
            raise ValueError(f"train target/option alignment failed for {row['id']}")
    train_indices = list(range(len(examples)))
    folds = assign_folds(examples, train_indices)
    components = connected_components(examples, train_indices)
    if (len(components) != config["data"].get("task_video_components")
            or sum(map(len, folds)) != len(train_indices)):
        raise ValueError("unexpected component count or fold coverage")

    context, option_rows, text_seconds = base._encode_text(
        examples, text_model, torch.device("cpu")
    )
    oof: dict[str, dict[int, dict[int, Tensor]]] = {
        mode: {epoch: {} for epoch in CURVE_EPOCHS} for mode in base.MODES
    }
    histories: dict[str, list[dict[str, Any]]] = {mode: [] for mode in base.MODES}
    fold_records = []
    training_started = time.perf_counter()

    for fold_id, heldout in enumerate(folds):
        heldout_set = set(heldout)
        fold_train = sorted(set(train_indices) - heldout_set)
        train_groups = {examples[i]["sid"] for i in fold_train}
        heldout_groups = {examples[i]["sid"] for i in heldout}
        train_videos = {examples[i]["video_sha256"] for i in fold_train}
        heldout_videos = {examples[i]["video_sha256"] for i in heldout}
        if train_groups & heldout_groups or train_videos & heldout_videos:
            raise ValueError(f"fold {fold_id} leaks task/video identity")
        sensor_mean, sensor_std = _fold_train_normalizer(arrays, fold_train)
        fold_records.append({
            "fold": fold_id,
            "train_rows": len(fold_train),
            "heldout_rows": len(heldout),
            "train_task_groups": len(train_groups),
            "heldout_task_groups": len(heldout_groups),
            "train_videos": len(train_videos),
            "heldout_videos": len(heldout_videos),
            "heldout_sample_ids": [examples[i]["id"] for i in heldout],
        })
        for mode in base.MODES:
            torch.manual_seed(base.SEED)
            model = base._new_model(mode)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=base.LEARNING_RATE,
                weight_decay=base.WEIGHT_DECAY,
            )
            rng = random.Random(base.SEED)
            for epoch in range(1, EPOCHS + 1):
                model.train()
                order = fold_train.copy()
                rng.shuffle(order)
                batch_losses = []
                for start in range(0, len(order), base.BATCH_SIZE):
                    batch_indices = order[start:start + base.BATCH_SIZE]
                    batch = base._batch(
                        batch_indices, examples, arrays, context, option_rows,
                        sensor_mean, sensor_std, mode, shuffle_options=True, rng=rng,
                    )
                    optimizer.zero_grad(set_to_none=True)
                    logits, targets = base._forward(model, mode, batch)
                    loss = F.cross_entropy(logits.float(), targets)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(
                            f"nonfinite loss at fold {fold_id} {mode} e{epoch}"
                        )
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    if not torch.isfinite(grad_norm):
                        raise FloatingPointError(
                            f"nonfinite gradient at fold {fold_id} {mode} e{epoch}"
                        )
                    optimizer.step()
                    batch_losses.append(float(loss.detach()))

                if epoch in CURVE_EPOCHS:
                    fold_predictions = predict_logits(
                        model, mode, heldout, examples, arrays, context, option_rows,
                        sensor_mean, sensor_std,
                    )
                    oof[mode][epoch].update(fold_predictions)
                histories[mode].append({
                    "fold": fold_id,
                    "epoch": epoch,
                    "train_rows": len(fold_train),
                    "optimizer_updates": epoch * ((len(fold_train) + base.BATCH_SIZE - 1)
                                                   // base.BATCH_SIZE),
                    "train_mean_batch_cross_entropy": sum(batch_losses) / len(batch_losses),
                })
            del model, optimizer

    training_seconds = time.perf_counter() - training_started
    metrics = {
        mode: {str(epoch): evaluate_oof(oof[mode][epoch], train_indices, examples)
               for epoch in sorted(CURVE_EPOCHS)}
        for mode in base.MODES
    }
    payload = {
        "status": "TRAIN_ONLY_GROUPED_OOF_COMPLETE",
        "experiment_id": experiment_id,
        "started_at_utc": started_at,
        "source_revision": "bae97d251907743f8fb6db7e70c61d361df4dc1e",
        "package_report_sha256": base.sha256_file(package_report_path),
        "transition_examples_sha256": base.sha256_file(examples_path),
        "feature_cache_sha256": base.sha256_file(feature_path),
        "config_sha256": __import__("hashlib").sha256(config_bytes).hexdigest(),
        "script_sha256": base.sha256_file(Path(__file__)),
        "seed": base.SEED,
        "fold_count": len(folds),
        "component_count": len(components),
        "train_examples": len(train_indices),
        "train_task_groups": len({examples[i]["sid"] for i in train_indices}),
        "train_videos": len({examples[i]["video_sha256"] for i in train_indices}),
        "epochs": EPOCHS,
        "curve_epochs": sorted(CURVE_EPOCHS),
        "modes": list(base.MODES),
        "checkpoint_selection": "none",
        "early_stopping": False,
        "training_performed": True,
        "development_validation_metrics_computed": False,
        "sealed_audit_accessed": False,
        "folds": fold_records,
        "text_feature_extraction_seconds": text_seconds,
        "training_seconds": training_seconds,
        "total_optimizer_updates": sum(
            item["optimizer_updates"] for rows in histories.values() for item in rows
            if item["epoch"] == EPOCHS
        ),
        "learning_curves_by_mode": histories,
        "out_of_fold_metrics_by_mode_and_epoch": metrics,
        "interpretation": (
            "Development diagnosis on HABIT's original train split only. OOF folds "
            "are grouped by connected task/video identity. No existing development "
            "validation or sealed audit metrics were used. This single-seed fixed "
            "schedule is not product-quality evidence or a blind benchmark."
        ),
    }
    (output / "report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (output / "folds.json").write_text(json.dumps(fold_records, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--text-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.package_dir, args.text_model, args.output, args.config)
    print(json.dumps({
        "status": report["status"],
        "fold_count": report["fold_count"],
        "component_count": report["component_count"],
        "train_examples": report["train_examples"],
        "train_task_groups": report["train_task_groups"],
        "train_videos": report["train_videos"],
        "training_seconds": report["training_seconds"],
        "total_optimizer_updates": report["total_optimizer_updates"],
    }, indent=2))


if __name__ == "__main__":
    main()
