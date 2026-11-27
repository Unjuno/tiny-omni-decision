"""Compare one-shard-sized and full-data frozen-feature training budgets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from scripts.audit_videoprism_checkpoint import CHECKPOINT_SHA256, MODEL_REVISION
from scripts.extract_frozen_videoprism_features import SOURCE_COMMIT
from scripts.train_intphys_next_feature_predictor import (
    load_feature_split,
    sha256_file,
    validate_group_disjointness,
)

DEFAULT_SEED = 17
DEFAULT_SMALL_TRAIN_COUNT = 3_004
DEFAULT_UPDATE_BUDGET = 3_901
DEFAULT_EVAL_INTERVAL = 200
DEFAULT_BATCH_SIZE = 64
DEFAULT_TRAIN_MONITOR_COUNT = 256
_SUBSET_DOMAIN = b"intphys-data-coverage-subset-v1:"


def deterministic_subset_indices(
    sample_ids: list[str], count: int, *, seed: int = DEFAULT_SEED
) -> list[int]:
    """Choose an ID-hash-ranked, label-blind subset independent of file order."""
    if not sample_ids or len(sample_ids) != len(set(sample_ids)):
        raise ValueError("sample IDs must be nonempty and unique")
    if type(count) is not int or count < 1 or count > len(sample_ids):
        raise ValueError("subset count must be in [1, number of sample IDs]")
    prefix = _SUBSET_DOMAIN + str(seed).encode() + b":"
    return sorted(
        range(len(sample_ids)),
        key=lambda index: (
            hashlib.sha256(prefix + sample_ids[index].encode("utf-8")).digest(),
            sample_ids[index],
        ),
    )[:count]


def _unique_group_representative_indices(
    sample_ids: list[str], group_ids: list[str]
) -> list[int]:
    """Select one deterministic sample per underlying group, independent of row order."""
    if len(sample_ids) != len(group_ids) or not sample_ids:
        raise ValueError("sample and group IDs must be matching nonempty lists")
    if len(set(sample_ids)) != len(sample_ids) or any(not value for value in group_ids):
        raise ValueError("sample IDs must be unique and group IDs nonempty")
    representatives: dict[str, tuple[str, int]] = {}
    for index, (sample_id, group_id) in enumerate(zip(sample_ids, group_ids, strict=True)):
        prior = representatives.get(group_id)
        if prior is None or sample_id < prior[0]:
            representatives[group_id] = (sample_id, index)
    return [index for _, index in sorted(representatives.values())]


def _identity_group_consistent(identities: list[str], group_ids: list[str]) -> bool:
    if len(identities) != len(group_ids):
        return False
    owners: dict[str, str] = {}
    for identity, group_id in zip(identities, group_ids, strict=True):
        if not identity or not group_id:
            return False
        prior = owners.setdefault(identity, group_id)
        if prior != group_id:
            return False
    return True


def _mean_loss(model: Any, features: Any, batch_size: int) -> float:
    import torch

    from tiny_omni_decision.temporal_prediction import cosine_prediction_loss

    model.eval()
    weighted_loss = 0.0
    count = 0
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            batch = features[start : start + batch_size]
            loss = cosine_prediction_loss(model(batch), batch)
            weighted_loss += float(loss) * len(batch)
            count += len(batch)
    if count == 0:
        raise ValueError("cannot evaluate an empty feature split")
    return weighted_loss / count


def _per_example_loss(model: Any, features: Any, batch_size: int) -> np.ndarray:
    import torch

    from tiny_omni_decision.temporal_prediction import per_sequence_cosine_error

    model.eval()
    rows: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            batch = features[start : start + batch_size]
            rows.append(per_sequence_cosine_error(model(batch), batch).cpu().numpy())
    if not rows:
        raise ValueError("cannot calculate per-example loss for an empty split")
    return np.concatenate(rows)


def _take_batch(
    train: Any, order: Any, cursor: int, batch_size: int, generator: Any
) -> tuple[Any, Any, Any, int]:
    """Consume one fixed-size batch from successive shuffled permutations."""
    import torch

    parts: list[Any] = []
    remaining = batch_size
    while remaining:
        if cursor == len(order):
            order = torch.randperm(len(train), generator=generator)
            cursor = 0
        take = min(remaining, len(order) - cursor)
        parts.append(order[cursor : cursor + take])
        cursor += take
        remaining -= take
    indices = parts[0] if len(parts) == 1 else torch.cat(parts)
    return train[indices], indices, order, cursor


def fit_fixed_update_budget(
    train_features: np.ndarray,
    validation_features: np.ndarray,
    train_monitor_features: np.ndarray,
    *,
    seed: int = DEFAULT_SEED,
    update_budget: int = DEFAULT_UPDATE_BUDGET,
    eval_interval: int = DEFAULT_EVAL_INTERVAL,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[Any, Any, list[dict[str, float]], int, float]:
    """Fit equal-update candidates and select the raw minimum fixed-validation loss."""
    import torch

    from tiny_omni_decision.temporal_prediction import (
        NextFrameFeaturePredictor,
        cosine_prediction_loss,
    )

    if type(update_budget) is not int or update_budget < 1:
        raise ValueError("update budget must be a positive integer")
    if type(eval_interval) is not int or eval_interval < 1:
        raise ValueError("evaluation interval must be a positive integer")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch size must be a positive integer")
    for name, array in (
        ("train", train_features),
        ("validation", validation_features),
        ("train monitor", train_monitor_features),
    ):
        if array.ndim != 3 or array.shape[0] < 1 or array.shape[1] < 2 or array.shape[2] < 1:
            raise ValueError(f"{name} features must be nonempty [N,T,D]")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} features must be finite")
    if train_features.shape[1:] != validation_features.shape[1:]:
        raise ValueError("train and validation feature dimensions must match")
    if train_monitor_features.shape[1:] != train_features.shape[1:]:
        raise ValueError("fixed train monitor dimensions must match training features")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    train = torch.from_numpy(np.ascontiguousarray(train_features))
    validation = torch.from_numpy(np.ascontiguousarray(validation_features))
    train_monitor = torch.from_numpy(np.ascontiguousarray(train_monitor_features))
    model = NextFrameFeaturePredictor(feature_width=train.shape[-1], hidden_width=64)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(train), generator=generator)
    cursor = 0
    seen_indices: set[int] = set()
    recent_losses: list[float] = []
    history: list[dict[str, float]] = []
    best_loss = float("inf")
    best_step = 0
    best_state: dict[str, torch.Tensor] | None = None
    started = time.perf_counter()

    for step in range(1, update_budget + 1):
        model.train()
        batch, batch_indices, order, cursor = _take_batch(
            train, order, cursor, batch_size, generator
        )
        seen_indices.update(int(index) for index in batch_indices.tolist())
        optimizer.zero_grad(set_to_none=True)
        loss = cosine_prediction_loss(model(batch), batch)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite training loss at update {step}")
        loss.backward()
        gradients = [
            parameter.grad for parameter in model.parameters() if parameter.grad is not None
        ]
        if not gradients or any(not torch.isfinite(gradient).all() for gradient in gradients):
            raise FloatingPointError(f"non-finite or missing gradients at update {step}")
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
        optimizer.step()
        recent_losses.append(float(loss.detach()))
        if len(recent_losses) > eval_interval:
            recent_losses.pop(0)

        if step % eval_interval == 0 or step == update_budget:
            validation_loss = _mean_loss(model, validation, batch_size)
            train_monitor_loss = _mean_loss(model, train_monitor, batch_size)
            row = {
                "optimizer_update": float(step),
                "learning_rate": 1e-3,
                "rolling_update_train_cosine_loss": float(np.mean(recent_losses)),
                "fixed_train_monitor_cosine_loss": train_monitor_loss,
                "validation_cosine_loss": validation_loss,
                "unique_train_examples_consumed": float(len(seen_indices)),
                "train_example_presentations": float(step * batch_size),
                "unique_underlying_groups_consumed": float(len(seen_indices)),
                "elapsed_seconds": time.perf_counter() - started,
            }
            history.append(row)
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_step = step
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }

    if best_state is None or best_step == 0:
        raise RuntimeError("fixed-update run produced no validation-selected checkpoint")
    terminal_state = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }
    return best_state, terminal_state, history, best_step, best_loss


def _paired_bootstrap_mean_difference(
    first: np.ndarray, second: np.ndarray, *, seed: int, replicates: int = 10_000
) -> dict[str, Any]:
    if first.shape != second.shape or first.ndim != 1 or len(first) == 0:
        raise ValueError("paired validation loss arrays must be matching nonempty vectors")
    differences = first - second
    generator = np.random.default_rng(seed)
    sampled = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sampled[index] = np.mean(
            differences[generator.integers(0, len(differences), len(differences))]
        )
    return {
        "small_minus_full_mean_validation_cosine_loss": float(np.mean(differences)),
        "small_minus_full_percentile_ci95": [
            float(np.quantile(sampled, 0.025)),
            float(np.quantile(sampled, 0.975)),
        ],
        "replicates": replicates,
        "seed": seed,
        "unit": "validation video group; identity groups are unique after global merge",
    }


def _save_candidate(
    name: str,
    state: Any,
    terminal_state: Any,
    history: list[dict[str, float]],
    best_step: int,
    best_loss: float,
    train_features: np.ndarray,
    train_ids: list[str],
    train_groups: list[str],
    validation_features: np.ndarray,
    validation_ids: list[str],
    validation_groups: list[str],
    validation_path: Path,
    output_dir: Path,
    *,
    started_at_utc: str,
    seed: int,
    batch_size: int,
    update_budget: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    import torch

    from tiny_omni_decision.temporal_prediction import NextFrameFeaturePredictor

    output_dir.mkdir(parents=True, exist_ok=False)
    model = NextFrameFeaturePredictor(feature_width=train_features.shape[-1], hidden_width=64)
    model.load_state_dict(state, strict=True)
    best_path = output_dir / "best-next-feature-predictor.pt"
    torch.save(
        {
            "state_dict": state,
            "feature_width": train_features.shape[-1],
            "hidden_width": 64,
            "seed": seed,
            "best_update": best_step,
        },
        best_path,
    )
    terminal_path = output_dir / "terminal-next-feature-predictor.pt"
    torch.save(
        {
            "state_dict": terminal_state,
            "feature_width": train_features.shape[-1],
            "hidden_width": 64,
            "seed": seed,
            "terminal_update": update_budget,
        },
        terminal_path,
    )
    history_path = output_dir / "training-history.jsonl"
    history_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in history),
        encoding="utf-8",
    )
    validation_errors = _per_example_loss(
        model, torch.from_numpy(np.ascontiguousarray(validation_features)), batch_size
    )
    prediction_path = output_dir / "validation-per-example.jsonl"
    prediction_path.write_text(
        "".join(
            json.dumps(
                {
                    "sample_id": sample_id,
                    "group_id": group_id,
                    "cosine_prediction_error": float(error),
                },
                sort_keys=True,
            )
            + "\n"
            for sample_id, group_id, error in zip(
                validation_ids, validation_groups, validation_errors, strict=True
            )
        ),
        encoding="utf-8",
    )
    parameters = sum(parameter.numel() for parameter in model.parameters())
    metadata = {
        "status": "complete_intphys_fixed_update_data_coverage_candidate",
        "candidate": name,
        "started_at_utc": started_at_utc,
        "ended_at_utc": datetime.now(UTC).isoformat(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "device": "cpu",
        "seed": seed,
        "optimizer": "AdamW",
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": batch_size,
        "update_budget": update_budget,
        "training_elapsed_seconds": history[-1]["elapsed_seconds"],
        "completed_updates": history[-1]["optimizer_update"],
        "selected_update": best_step,
        "checkpoint_selection": "minimum raw validation cosine loss at fixed evaluation points",
        "train_unique_examples": len(train_features),
        "train_unique_groups": len(set(train_groups)),
        "train_presentations": update_budget * batch_size,
        "unique_train_examples_consumed": int(
            history[-1]["unique_train_examples_consumed"]
        ),
        "train_repeated_presentations": max(
            0,
            update_budget * batch_size
            - int(history[-1]["unique_train_examples_consumed"]),
        ),
        "train_sample_ids_sha256": hashlib.sha256(
            "\n".join(train_ids).encode()
        ).hexdigest(),
        "validation_examples": len(validation_features),
        "validation_unique_groups": len(set(validation_groups)),
        "validation_feature_sha256": sha256_file(validation_path),
        "validation_ids_sha256": hashlib.sha256("\n".join(validation_ids).encode()).hexdigest(),
        "selected_validation_cosine_loss": best_loss,
        "selected_checkpoint_sha256": sha256_file(best_path),
        "terminal_checkpoint_sha256": sha256_file(terminal_path),
        "history_sha256": sha256_file(history_path),
        "validation_predictions_sha256": sha256_file(prediction_path),
        "trainable_parameter_count": parameters,
        "quality_scope": (
            "frozen-feature next-frame prediction objective only; not Decision accuracy "
            "or calibrated probability"
        ),
    }
    reloaded = NextFrameFeaturePredictor(feature_width=train_features.shape[-1], hidden_width=64)
    checkpoint = torch.load(best_path, map_location="cpu", weights_only=True)
    reloaded.load_state_dict(checkpoint["state_dict"], strict=True)
    reloaded_loss = _mean_loss(
        reloaded,
        torch.from_numpy(np.ascontiguousarray(validation_features)),
        batch_size,
    )
    if not math.isclose(reloaded_loss, best_loss, rel_tol=0.0, abs_tol=1e-8):
        raise RuntimeError("reloaded best checkpoint does not reproduce selected validation loss")
    metadata["best_reload_validation_cosine_loss"] = reloaded_loss
    (output_dir / "run-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    metadata["run_metadata_sha256"] = sha256_file(output_dir / "run-metadata.json")
    return validation_errors, metadata


def run_comparison(
    train_path: Path,
    validation_path: Path,
    output_dir: Path,
    *,
    seed: int = DEFAULT_SEED,
    small_train_count: int = DEFAULT_SMALL_TRAIN_COUNT,
    update_budget: int = DEFAULT_UPDATE_BUDGET,
    eval_interval: int = DEFAULT_EVAL_INTERVAL,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing attempt: {output_dir}")
    train_path = train_path.resolve(strict=True)
    validation_path = validation_path.resolve(strict=True)
    if train_path == validation_path:
        raise ValueError("training and validation cache paths must differ")
    merge_report_path = train_path.parent / "merge-report.json"
    if not merge_report_path.is_file():
        raise FileNotFoundError(f"missing global shard-merge receipt: {merge_report_path}")
    merge_report = json.loads(merge_report_path.read_text(encoding="utf-8"))
    if merge_report.get("status") != "complete_intphys_all_train_shards_globally_grouped":
        raise ValueError("feature caches are not backed by a complete global shard merge")
    if merge_report.get("required_shards") != [1, 2, 3, 4]:
        raise ValueError("global merge receipt does not include all four official train shards")
    if (
        merge_report.get("frame_count") != 8
        or merge_report.get("feature_shape_per_sample") != [8, 768]
        or merge_report.get("model_revision") != MODEL_REVISION
        or merge_report.get("videoprism_source_commit") != SOURCE_COMMIT
        or merge_report.get("checkpoint_sha256") != CHECKPOINT_SHA256
    ):
        raise ValueError("global merge receipt differs from the pinned 8-frame VideoPrism path")
    if merge_report.get("feature_archive_sha256") != {
        "train": sha256_file(train_path),
        "validation": sha256_file(validation_path),
    }:
        raise ValueError("global merge receipt does not bind the supplied feature caches")
    for overlap_key in (
        "cross_split_group_overlap",
        "cross_split_status_identity_overlap",
        "cross_split_rgb_sequence_identity_overlap",
        "cross_split_sampled_rgb_input_overlap",
    ):
        if merge_report.get(overlap_key) != 0:
            raise ValueError(f"global shard-merge receipt fails {overlap_key}")
    train_rows, train_row_ids, train_row_groups = load_feature_split(train_path)
    validation_rows, validation_row_ids, validation_row_groups = load_feature_split(
        validation_path
    )
    if train_rows.shape[1:] != (8, 768) or validation_rows.shape[1:] != (8, 768):
        raise ValueError("data-coverage candidates require pinned [N, 8, 768] features")
    with np.load(train_path, allow_pickle=False) as cache:
        train_identities = {
            key: cache[key].astype(str).tolist()
            for key in (
                "status_identity_sha256",
                "rgb_sequence_sha256",
                "sampled_rgb_pixels_sha256",
            )
        }
    with np.load(validation_path, allow_pickle=False) as cache:
        validation_identities = {
            key: cache[key].astype(str).tolist()
            for key in (
                "status_identity_sha256",
                "rgb_sequence_sha256",
                "sampled_rgb_pixels_sha256",
            )
        }
    if any(len(values) != len(train_rows) for values in train_identities.values()):
        raise ValueError("global training cache identity arrays do not match the features")
    if any(len(values) != len(validation_rows) for values in validation_identities.values()):
        raise ValueError("global validation cache identity arrays do not match the features")
    validate_group_disjointness(
        train_row_ids, train_row_groups, validation_row_ids, validation_row_groups
    )
    for name, train_values in train_identities.items():
        validation_values = validation_identities[name]
        if not _identity_group_consistent(train_values, train_row_groups):
            raise ValueError(f"training identity {name} maps to multiple global groups")
        if not _identity_group_consistent(validation_values, validation_row_groups):
            raise ValueError(f"validation identity {name} maps to multiple global groups")
        if set(train_values) & set(validation_values):
            raise ValueError(f"underlying media identity overlap for {name}")

    train_representatives = _unique_group_representative_indices(
        train_row_ids, train_row_groups
    )
    validation_representatives = _unique_group_representative_indices(
        validation_row_ids, validation_row_groups
    )
    train = train_rows[train_representatives]
    train_ids = [train_row_ids[index] for index in train_representatives]
    train_groups = [train_row_groups[index] for index in train_representatives]
    validation = validation_rows[validation_representatives]
    validation_ids = [validation_row_ids[index] for index in validation_representatives]
    validation_groups = [validation_row_groups[index] for index in validation_representatives]
    train_group_identity_rows_removed = len(train_rows) - len(train)
    validation_group_identity_rows_removed = len(validation_rows) - len(validation)
    if small_train_count > len(train):
        raise ValueError("small-data count exceeds the globally unique training groups")
    subset_indices = deterministic_subset_indices(train_ids, small_train_count, seed=seed)
    subset_features = train[subset_indices]
    subset_ids = [train_ids[index] for index in subset_indices]
    subset_groups = [train_groups[index] for index in subset_indices]
    monitor_indices = deterministic_subset_indices(
        subset_ids, min(DEFAULT_TRAIN_MONITOR_COUNT, len(subset_ids)), seed=seed + 1
    )
    monitor_features = subset_features[monitor_indices]
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing attempt: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    subset_path = output_dir / "small-data-subset-sample-ids.jsonl"
    subset_path.write_text(
        "".join(
            json.dumps({"sample_id": sample_id, "group_id": group_id}, sort_keys=True) + "\n"
            for sample_id, group_id in zip(subset_ids, subset_groups, strict=True)
        ),
        encoding="utf-8",
    )

    run_results: dict[str, Any] = {}
    val_errors: dict[str, np.ndarray] = {}
    small_candidate_name = f"D-small-{small_train_count}"
    for name, candidate_features, candidate_ids, candidate_groups in (
        (small_candidate_name, subset_features, subset_ids, subset_groups),
        ("D-full", train, train_ids, train_groups),
    ):
        candidate_started_at_utc = datetime.now(UTC).isoformat()
        best_state, terminal_state, history, best_step, best_loss = fit_fixed_update_budget(
            candidate_features,
            validation,
            monitor_features,
            seed=seed,
            update_budget=update_budget,
            eval_interval=eval_interval,
            batch_size=batch_size,
        )
        errors, metadata = _save_candidate(
            name,
            best_state,
            terminal_state,
            history,
            best_step,
            best_loss,
            candidate_features,
            candidate_ids,
            candidate_groups,
            validation,
            validation_ids,
            validation_groups,
            validation_path,
            output_dir / name,
            started_at_utc=candidate_started_at_utc,
            seed=seed,
            batch_size=batch_size,
            update_budget=update_budget,
        )
        val_errors[name] = errors
        run_results[name] = metadata
        del best_state, terminal_state, history

    comparison = {
        "schema_version": 1,
        "status": "complete_intphys_data_coverage_comparison",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "command": sys.argv,
        "seed": seed,
        "train_feature_path": str(train_path),
        "train_feature_sha256": sha256_file(train_path),
        "validation_feature_path": str(validation_path),
        "validation_feature_sha256": sha256_file(validation_path),
        "global_merge_report_path": str(merge_report_path),
        "global_merge_report_sha256": sha256_file(merge_report_path),
        "train_examples_global": len(train),
        "train_rows_before_group_deduplication": len(train_rows),
        "train_duplicate_identity_rows_excluded": train_group_identity_rows_removed,
        "validation_examples_global": len(validation),
        "validation_rows_before_group_deduplication": len(validation_rows),
        "validation_duplicate_identity_rows_excluded": validation_group_identity_rows_removed,
        "train_unique_groups_global": len(set(train_groups)),
        "validation_unique_groups_global": len(set(validation_groups)),
        "shared_validation_ids_sha256": hashlib.sha256(
            "\n".join(validation_ids).encode()
        ).hexdigest(),
        "small_subset_count": small_train_count,
        "small_subset_ids_sha256": hashlib.sha256("\n".join(subset_ids).encode()).hexdigest(),
        "small_subset_selection": (
            "ascending SHA-256(seed 17, sample ID), label-blind and independent of source order"
        ),
        "optimizer_update_budget": update_budget,
        "evaluation_interval": eval_interval,
        "batch_size": batch_size,
        "unique_assets": {
            small_candidate_name: len(set(subset_groups)),
            "D-full": len(set(train_groups)),
        },
        "selected_validation_cosine_loss": {
            name: run_results[name]["selected_validation_cosine_loss"] for name in run_results
        },
        "selected_step": {name: run_results[name]["selected_update"] for name in run_results},
        "paired_validation_loss_diagnostic": _paired_bootstrap_mean_difference(
            val_errors[small_candidate_name], val_errors["D-full"], seed=seed
        ),
        "candidates": run_results,
        "dev_or_sealed_audit_accessed": False,
        "quality_scope": (
            "internal source-held-out next-feature prediction only; not a physical-plausibility "
            "or general Decision evaluation"
        ),
    }
    path = output_dir / "comparison-report.json"
    path.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return comparison


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-features", type=Path, required=True)
    parser.add_argument("--validation-features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--small-train-count", type=int, default=DEFAULT_SMALL_TRAIN_COUNT)
    parser.add_argument("--update-budget", type=int, default=DEFAULT_UPDATE_BUDGET)
    parser.add_argument("--eval-interval", type=int, default=DEFAULT_EVAL_INTERVAL)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    return parser.parse_args()


def main() -> None:
    args = _args()
    result = run_comparison(
        args.train_features,
        args.validation_features,
        args.output_dir,
        seed=args.seed,
        small_train_count=args.small_train_count,
        update_budget=args.update_budget,
        eval_interval=args.eval_interval,
        batch_size=args.batch_size,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
