"""Run a train-only leave-one-scenario-out diagnostic for frozen VideoPrism Path A."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.nn import functional as F

from scripts.train_physionpp_videoprism_path_a import (
    CANDIDATE_DIMENSION,
    EXPECTED_TRAIN_HASH,
    VIDEO_DIMENSION,
    _batch_logits,
    _build_features,
    _read_jsonl,
    _targets,
    load_feature_cache,
    metric_values,
)
from tiny_omni_decision.dataset import sha256_file
from tiny_omni_decision.decision import FrozenFeatureCandidateScorer


def scenario_name(row: dict[str, Any]) -> str:
    group = row.get("source_asset_group_id")
    if not isinstance(group, str) or ":seed=" not in group:
        raise ValueError("source_asset_group_id must encode scenario:seed=...")
    scenario, seed = group.rsplit(":seed=", 1)
    if not scenario or not seed:
        raise ValueError("invalid scenario/seed group identity")
    return scenario


def leave_one_scenario_out(rows: list[dict[str, Any]]) -> dict[str, tuple[list[int], list[int]]]:
    scenarios = [scenario_name(row) for row in rows]
    unique_scenarios = sorted(set(scenarios))
    if len(unique_scenarios) < 2:
        raise ValueError("scenario holdout requires at least two scenarios")
    folds: dict[str, tuple[list[int], list[int]]] = {}
    for held_out in unique_scenarios:
        train_indices = [index for index, name in enumerate(scenarios) if name != held_out]
        held_indices = [index for index, name in enumerate(scenarios) if name == held_out]
        train_groups = {rows[index]["source_asset_group_id"] for index in train_indices}
        held_groups = {rows[index]["source_asset_group_id"] for index in held_indices}
        if not held_indices or train_groups & held_groups:
            raise ValueError(f"invalid scene-group split for held-out scenario {held_out}")
        folds[held_out] = (train_indices, held_indices)
    held_once = [index for _train, held in folds.values() for index in held]
    if sorted(held_once) != list(range(len(rows))):
        raise ValueError("each row must occur in exactly one held-out scenario fold")
    return folds


def _aggregate(probabilities: np.ndarray, targets: np.ndarray) -> dict[str, float | int]:
    logits = torch.from_numpy(np.log(np.clip(probabilities, 1e-12, 1.0))).float()
    return metric_values(logits, torch.from_numpy(targets).long())


def evaluate_train_scenarios(
    *,
    train_path: Path,
    cache_path: Path,
    cache_report_path: Path,
    config_path: Path,
    output: Path,
    bootstrap_replicates: int = 10_000,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    if sha256_file(train_path) != EXPECTED_TRAIN_HASH:
        raise ValueError("frozen train manifest hash mismatch")
    rows = _read_jsonl(train_path, "train")
    features = load_feature_cache(
        cache_path,
        cache_report_path,
        rows,
        expected_split="train",
        expected_manifest_sha256=EXPECTED_TRAIN_HASH,
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    readout_config = config["readout"]
    expected = {
        "seed": 17,
        "epochs": 200,
        "early_stopping": False,
        "learning_rate": 0.001,
        "weight_decay": 0.001,
        "batch_questions": 32,
        "hidden_size": 128,
    }
    effective = {
        "seed": int(readout_config["seed"]),
        "epochs": int(readout_config["epochs"]),
        "early_stopping": readout_config["early_stopping"],
        "learning_rate": float(readout_config["learning_rate"]),
        "weight_decay": float(readout_config["weight_decay"]),
        "batch_questions": int(readout_config["batch_questions"]),
        "hidden_size": int(readout_config["hidden_size"]),
    }
    if effective != expected:
        raise ValueError("scenario holdout must match the fixed Path A optimization schedule")

    targets = _targets(rows).numpy()
    folds = leave_one_scenario_out(rows)
    output.mkdir(parents=True, exist_ok=False)
    scenario_by_index = [scenario_name(row) for row in rows]
    out_of_fold: dict[str, list[float]] = {}
    zero_video_oof: dict[str, list[float]] = {}
    permuted_video_oof: dict[str, list[float]] = {}
    prior_oof: dict[str, list[float]] = {}
    scenario_reports: list[dict[str, Any]] = []
    fold_histories: dict[str, list[float]] = {}
    parameter_count: int | None = None
    total_training_seconds = 0.0
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    torch.set_num_threads(4)

    for fold_index, (scenario, (train_indices, held_indices)) in enumerate(folds.items()):
        seed = int(expected["seed"])
        torch.manual_seed(seed)
        scorer = FrozenFeatureCandidateScorer(
            VIDEO_DIMENSION + CANDIDATE_DIMENSION, int(expected["hidden_size"])
        )
        count = sum(parameter.numel() for parameter in scorer.parameters())
        if count != int(readout_config["trainable_parameters_expected"]):
            raise ValueError("trainable parameter count mismatch")
        parameter_count = count
        optimizer = torch.optim.AdamW(
            scorer.parameters(),
            lr=float(expected["learning_rate"]),
            weight_decay=float(expected["weight_decay"]),
        )
        generator = torch.Generator().manual_seed(seed)
        train_rows = [rows[index] for index in train_indices]
        held_rows = [rows[index] for index in held_indices]
        train_videos = features[train_indices]
        held_videos = features[held_indices]
        train_features = _build_features(train_videos, train_rows)
        held_features = _build_features(held_videos, held_rows)
        train_targets = torch.from_numpy(targets[train_indices]).long()
        held_targets = torch.from_numpy(targets[held_indices]).long()
        losses: list[float] = []
        started = time.perf_counter()

        for _epoch in range(1, int(expected["epochs"]) + 1):
            scorer.train()
            order = torch.randperm(len(train_features), generator=generator)
            epoch_losses = []
            for start in range(0, len(order), int(expected["batch_questions"])):
                indices = order[start : start + int(expected["batch_questions"])]
                batch = train_features[indices]
                batch_targets = train_targets[indices]
                swaps = torch.rand(len(indices), generator=generator) < 0.5
                first = torch.where(swaps, 1, 0)
                second = 1 - first
                permutation = torch.stack([first, second], dim=1)
                gather_index = permutation.unsqueeze(-1).expand(-1, -1, batch.shape[-1])
                batch = batch.gather(1, gather_index)
                batch_targets = torch.where(swaps, 1 - batch_targets, batch_targets)
                logits = scorer(batch)
                loss = F.cross_entropy(logits.float(), batch_targets)
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"non-finite loss in {scenario}, epoch {_epoch}"
                    )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                epoch_losses.append(float(loss.detach()))
            losses.append(float(np.mean(epoch_losses)))

        fold_seconds = time.perf_counter() - started
        total_training_seconds += fold_seconds
        fold_histories[scenario] = losses
        held_logits = _batch_logits(scorer, held_features)
        held_probabilities = torch.softmax(held_logits, dim=-1).numpy()
        for local_index, global_index in enumerate(held_indices):
            out_of_fold[rows[global_index]["id"]] = held_probabilities[local_index].tolist()

        zero_videos = held_videos.copy()
        zero_videos[:] = 0.0
        zero_logits = _batch_logits(scorer, _build_features(zero_videos, held_rows))
        zero_probabilities = torch.softmax(zero_logits, dim=-1).numpy()
        for local_index, global_index in enumerate(held_indices):
            zero_video_oof[rows[global_index]["id"]] = zero_probabilities[
                local_index
            ].tolist()

        permutation = np.random.default_rng(17017 + fold_index).permutation(
            len(held_videos)
        )
        permuted_features = _build_features(held_videos[permutation], held_rows)
        permuted_logits = _batch_logits(scorer, permuted_features)
        permuted_probabilities = torch.softmax(permuted_logits, dim=-1).numpy()
        for local_index, global_index in enumerate(held_indices):
            permuted_video_oof[rows[global_index]["id"]] = permuted_probabilities[
                local_index
            ].tolist()

        fold_prior = np.bincount(targets[train_indices], minlength=2) / len(train_indices)
        for global_index in held_indices:
            prior_oof[rows[global_index]["id"]] = fold_prior.tolist()

        scenario_reports.append(
            {
                "held_out_scenario": scenario,
                "train_records": len(train_indices),
                "held_out_records": len(held_indices),
                "train_groups": len(
                    {rows[index]["source_asset_group_id"] for index in train_indices}
                ),
                "held_out_groups": len(
                    {rows[index]["source_asset_group_id"] for index in held_indices}
                ),
                "train_label_counts": {
                    "No": int((targets[train_indices] == 0).sum()),
                    "Yes": int((targets[train_indices] == 1).sum()),
                },
                "held_out_label_counts": {
                    "No": int((targets[held_indices] == 0).sum()),
                    "Yes": int((targets[held_indices] == 1).sum()),
                },
                "held_out_metrics": metric_values(held_logits, held_targets),
                "zero_video_metrics": metric_values(zero_logits, held_targets),
                "permuted_video_metrics": metric_values(permuted_logits, held_targets),
                "permutation_seed": 17017 + fold_index,
                "training_seconds": fold_seconds,
            }
        )
        checkpoint_path = output / "fold-checkpoints" / f"{fold_index:02d}-{scenario}.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in scorer.state_dict().items()
                },
                "held_out_scenario": scenario,
                "train_ids_sha256": hashlib.sha256(
                    "\n".join(rows[index]["id"] for index in train_indices).encode()
                ).hexdigest(),
                "seed": seed,
                "epochs": int(expected["epochs"]),
            },
            checkpoint_path,
        )

    ordered_ids = [row["id"] for row in rows]
    probabilities = np.asarray([out_of_fold[sample_id] for sample_id in ordered_ids])
    zero_probabilities = np.asarray([zero_video_oof[sample_id] for sample_id in ordered_ids])
    permuted_probabilities = np.asarray(
        [permuted_video_oof[sample_id] for sample_id in ordered_ids]
    )
    prior_probabilities = np.asarray([prior_oof[sample_id] for sample_id in ordered_ids])
    oof_metrics = _aggregate(probabilities, targets)
    zero_metrics = _aggregate(zero_probabilities, targets)
    permuted_metrics = _aggregate(permuted_probabilities, targets)
    prior_metrics = _aggregate(prior_probabilities, targets)
    by_scenario_accuracy = [
        float(item["held_out_metrics"]["accuracy"]) for item in scenario_reports
    ]
    macro_scenario_accuracy = float(np.mean(by_scenario_accuracy))

    unique_groups = sorted({row["source_asset_group_id"] for row in rows})
    group_indices = {
        group: np.asarray(
            [index for index, row in enumerate(rows) if row["source_asset_group_id"] == group],
            dtype=np.int64,
        )
        for group in unique_groups
    }
    bootstrap_rng = np.random.default_rng(20261128)
    paired_differences = np.empty((bootstrap_replicates, 3), dtype=np.float64)

    def three_metrics(probs: np.ndarray, indices: np.ndarray) -> np.ndarray:
        selected = np.clip(probs[indices], 1e-12, 1.0)
        labels = targets[indices]
        return np.asarray(
            [
                (selected.argmax(axis=1) == labels).mean(),
                -np.log(selected[np.arange(len(indices)), labels]).mean(),
                ((selected - np.eye(2)[labels]) ** 2).sum(axis=1).mean(),
            ],
            dtype=np.float64,
        )

    for replicate in range(bootstrap_replicates):
        sampled_groups = bootstrap_rng.choice(unique_groups, size=len(unique_groups), replace=True)
        sample_indices = np.concatenate([group_indices[group] for group in sampled_groups])
        paired_differences[replicate] = three_metrics(
            probabilities, sample_indices
        ) - three_metrics(zero_probabilities, sample_indices)

    prediction_path = output / "out-of-scenario-predictions.jsonl"
    with prediction_path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            handle.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "scenario": scenario_by_index[index],
                        "source_asset_group_id": row["source_asset_group_id"],
                        "target": row["target"],
                        "out_of_scenario_probabilities": probabilities[index].tolist(),
                        "zero_video_probabilities": zero_probabilities[index].tolist(),
                        "permuted_video_probabilities": permuted_probabilities[index].tolist(),
                        "training_prior_probabilities": prior_probabilities[index].tolist(),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    history_path = output / "fold-training-losses.json"
    history_path.write_text(json.dumps(fold_histories, indent=2) + "\n", encoding="utf-8")
    pair_ci = np.quantile(paired_differences, [0.025, 0.5, 0.975], axis=0).tolist()
    report = {
        "schema_version": 1,
        "status": "complete_train_only_leave_one_scenario_out_diagnostic",
        "claim_scope": (
            "exploratory train-only scenario-held-out diagnostics; not independent, blind, "
            "publisher-test or product-quality evidence"
        ),
        "source_commit": subprocess_head(),
        "run_started_utc": started_utc,
        "run_finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "train_manifest_sha256": sha256_file(train_path),
        "feature_cache_sha256": sha256_file(cache_path),
        "feature_cache_report_sha256": sha256_file(cache_report_path),
        "config_sha256": sha256_file(config_path),
        "train_rows": len(rows),
        "source_asset_groups": len(unique_groups),
        "scenario_folds": len(folds),
        "held_out_scenario_records": sum(len(held) for _train, held in folds.values()),
        "held_out_scenarios": scenario_reports,
        "out_of_scenario_oof_metrics": oof_metrics,
        "macro_accuracy_across_scenarios": macro_scenario_accuracy,
        "minimum_scenario_accuracy": float(min(by_scenario_accuracy)),
        "zero_video_oof_metrics": zero_metrics,
        "permuted_video_oof_metrics": permuted_metrics,
        "training_prior_oof_metrics": prior_metrics,
        "paired_group_bootstrap_vs_zero_video": {
            "group_key": "source_asset_group_id (scenario plus scene seed)",
            "groups": len(unique_groups),
            "replicates": bootstrap_replicates,
            "seed": 20261128,
            "metric_order": ["accuracy", "nll", "brier"],
            "percentiles_2_5_50_97_5": pair_ci,
            "caveat": "descriptive train-only CV; not independent data or selection-bias corrected",
        },
        "effective_training": {
            "optimizer": "AdamW",
            "learning_rate": expected["learning_rate"],
            "weight_decay": expected["weight_decay"],
            "seed_per_fold": expected["seed"],
            "epochs_per_fold": expected["epochs"],
            "batch_questions": expected["batch_questions"],
            "early_stopping": False,
            "checkpoint_selection": "none; final fixed-budget fold checkpoint",
            "trainable_parameters": parameter_count,
            "total_fold_training_seconds": total_training_seconds,
            "device": "cpu",
            "gpu_used": False,
        },
        "prior_full_train_model_has_seen_these_records": True,
        "development_validation_loaded": False,
        "publisher_test_or_sealed_audit_loaded": False,
        "artifacts": {
            "predictions_sha256": sha256_file(prediction_path),
            "fold_history_sha256": sha256_file(history_path),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "pyyaml": importlib.metadata.version("PyYAML"),
        },
    }
    (output / "run-report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def subprocess_head() -> str:
    import subprocess

    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--train-feature-cache", type=Path, required=True)
    parser.add_argument("--train-feature-report", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/pretrained_reuse/path_a_video_videoprism_physionpp_contact_seed17.yaml"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    args = parser.parse_args()
    if args.bootstrap_replicates < 1:
        parser.error("--bootstrap-replicates must be positive")
    report = evaluate_train_scenarios(
        train_path=args.train.resolve(strict=True),
        cache_path=args.train_feature_cache.resolve(strict=True),
        cache_report_path=args.train_feature_report.resolve(strict=True),
        config_path=args.config.resolve(strict=True),
        output=args.output.resolve(),
        bootstrap_replicates=args.bootstrap_replicates,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
