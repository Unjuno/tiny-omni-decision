"""Evaluate fixed EgoViz readouts once on new, session-disjoint development groups."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def metrics(logits: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    one_hot = np.eye(logits.shape[1])[targets]
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == targets
    ece = 0.0
    for lower in np.linspace(0, 1, 11)[:-1]:
        upper = lower + 0.1
        upper_test = confidence < upper if upper < 1 else confidence <= upper
        selected = (confidence >= lower) & upper_test
        if selected.any():
            ece += selected.mean() * abs(correct[selected].mean() - confidence[selected].mean())
    return {
        "accuracy": float(correct.mean()),
        "nll": float(
            -np.log(np.maximum(probabilities[np.arange(len(targets)), targets], 1e-15)).mean()
        ),
        "brier": float(((probabilities - one_hot) ** 2).sum(axis=1).mean()),
        "ece": float(ece),
        "mean_confidence": float(confidence.mean()),
        "count": int(len(targets)),
    }


def grouped_metrics(
    logits: np.ndarray, targets: np.ndarray, categories: np.ndarray
) -> dict[str, Any]:
    logits = np.asarray(logits)
    targets = np.asarray(targets)
    categories = np.asarray(categories)
    if logits.ndim != 2 or len(logits) != len(targets) or len(targets) != len(categories):
        raise ValueError("logits, targets, and category arrays must align")
    by_category = {}
    for category in sorted(set(categories.tolist())):
        mask = categories == category
        by_category[str(category)] = metrics(logits[mask], targets[mask])
    return {"overall": metrics(logits, targets), "by_category": by_category}


def paired_group_bootstrap(
    candidate_logits: np.ndarray,
    reference_logits: np.ndarray,
    targets: np.ndarray,
    groups: np.ndarray,
    *,
    draws: int = 2_000,
    seed: int = 17,
) -> dict[str, Any]:
    candidate_logits = np.asarray(candidate_logits)
    reference_logits = np.asarray(reference_logits)
    targets = np.asarray(targets)
    groups = np.asarray(groups)
    if (
        candidate_logits.shape != reference_logits.shape
        or candidate_logits.ndim != 2
        or len(targets) != len(candidate_logits)
        or len(groups) != len(targets)
        or draws <= 0
    ):
        raise ValueError("paired bootstrap inputs do not align")
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("paired group bootstrap requires at least two groups")
    indices_by_group = {group: np.flatnonzero(groups == group) for group in unique_groups}
    deltas = {name: [] for name in ("accuracy", "nll", "brier", "ece")}
    rng = np.random.default_rng(seed)
    for _ in range(draws):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([indices_by_group[group] for group in sampled])
        candidate = metrics(candidate_logits[indices], targets[indices])
        reference = metrics(reference_logits[indices], targets[indices])
        for name in deltas:
            deltas[name].append(candidate[name] - reference[name])
    candidate_point = metrics(candidate_logits, targets)
    reference_point = metrics(reference_logits, targets)
    return {
        "method": "paired group_id bootstrap; percentile 95% interval",
        "draws": draws,
        "seed": seed,
        "group_count": int(len(unique_groups)),
        "candidate_minus_reference": {
            name: {
                "estimate": candidate_point[name] - reference_point[name],
                "percentile_95_ci": [
                    float(np.quantile(values, 0.025)),
                    float(np.quantile(values, 0.975)),
                ],
            }
            for name, values in deltas.items()
        },
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError(f"expected JSON object: {path}")
    return result


def validate_matched_training_provenance(
    linear: dict[str, Any], gru: dict[str, Any]
) -> None:
    """Require both fixed checkpoints to come from the same frozen run inputs."""
    if linear.get("seed") != gru.get("seed"):
        raise ValueError("linear and GRU training seeds differ")
    if linear.get("input_sha256") != gru.get("input_sha256"):
        raise ValueError("linear and GRU training input hashes differ")
    linear_order = linear.get("sample_order_sha256")
    gru_order = gru.get("training", {}).get("sample_order_sha256")
    if not linear_order or linear_order != gru_order:
        raise ValueError("linear and GRU training sample-order hashes differ")


def _load_checkpoint(model, path: Path, expected_sha256: str) -> None:
    import torch

    if sha256_file(path) != expected_sha256:
        raise ValueError(f"checkpoint hash mismatch: {path}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError(f"expected weights-only state dict: {path}")
    model.load_state_dict(state, strict=True)
    model.eval()


def _predict(model, data) -> np.ndarray:
    import torch

    with torch.inference_mode():
        logits = model(data[0], data[1]).detach().cpu().numpy().astype(np.float64)
    if logits.ndim != 2 or logits.shape != (len(data[4]), 4) or not np.isfinite(logits).all():
        raise ValueError("model returned invalid four-option logits")
    return logits


def evaluate(
    *,
    plan_path: Path,
    corpus: Path,
    linear_run: Path,
    gru_run: Path,
    output: Path,
    draws: int = 2_000,
) -> dict[str, Any]:
    import torch

    from scripts.train_egoviz_gru_temporal import OrderedCrossModalGRUScorer
    from scripts.train_egoviz_vs_sync_readout import (
        JointTemporalScorer,
        _load_samples,
        _tensor_data,
    )

    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    plan = _read_json(plan_path)
    aggregate = _read_json(corpus / "aggregate-report.json")
    linear = _read_json(linear_run / "results.json")
    gru = _read_json(gru_run / "results.json")
    if (
        plan.get("status") != "METADATA_ONLY_FRESH_DEVELOPMENT_PLAN_MEDIA_NOT_DOWNLOADED"
        or aggregate.get("status") != "FROZEN_FEATURES_AGGREGATED_NO_TRAINING_OR_QUALITY_EVALUATION"
        or aggregate.get("sealed_audit_accessed") is not False
        or aggregate.get("training") is not False
        or aggregate.get("development_sessions") != plan["totals"]["eligible_unseen_groups"]
        or linear.get("status") != "FIXED_COVERAGE_BUDGET_COMPLETE_EXPLORATORY"
        or linear.get("updates") != 5_632
        or linear.get("train_examples") != 773
        or linear.get("development_examples") != 307
        or linear.get("train_sessions") != 48
        or gru.get("status") != "FIXED_BUDGET_COMPLETE_EXPLORATORY_DEVELOPMENT"
        or gru.get("training", {}).get("updates") != 5_632
        or gru.get("training", {}).get("train_examples") != 773
        or gru.get("training", {}).get("development_examples") != 307
        or gru.get("training", {}).get("train_sessions") != 48
        or linear.get("sealed_audit_accessed") is not False
        or gru.get("sealed_audit_accessed") is not False
    ):
        raise ValueError("fresh-split or fixed-checkpoint provenance gate failed")
    validate_matched_training_provenance(linear, gru)

    for name, digest in aggregate["sha256"].items():
        if sha256_file(corpus / name) != digest:
            raise ValueError(f"fresh corpus hash mismatch: {name}")
    if linear.get("best_step") != 616:
        raise ValueError("linear historical selector step changed")
    linear_best_path = linear_run / "best.pt"
    linear_final_path = linear_run / "step-5632.pt"
    gru_final_path = gru_run / "step-5632.pt"

    torch.set_num_threads(4)
    rows, video, sensor = _load_samples(
        corpus, corpus / "video-features.npz", corpus / "sensor-features.npz"
    )
    development_rows = [row for row in rows if row["split"] == "development"]
    if len(development_rows) != aggregate.get("development_examples"):
        raise ValueError("fresh sample count differs from aggregate report")
    plan_groups = {str(row["group_id"]) for row in plan["splits"]["development"]}
    excluded_groups = {str(group) for group in plan["source"]["excluded_groups"]}
    observed_groups = {str(row["group_id"]) for row in development_rows}
    if (
        observed_groups != plan_groups
        or observed_groups & excluded_groups
        or len({row["sample_id"] for row in development_rows}) != len(development_rows)
    ):
        raise ValueError("fresh development sample/group overlap or duplicate identity")
    dev = _tensor_data(rows, video, sensor, "development")
    categories = np.asarray([str(row["category"]) for row in development_rows])
    groups = np.asarray(dev[3])
    targets = dev[2].numpy()
    if len(np.unique(groups)) != aggregate["development_sessions"]:
        raise ValueError("unexpected number of fresh development groups")

    model_specs = {
        "linear_step_5632": (
            JointTemporalScorer(),
            linear_final_path,
            linear["final_checkpoint_sha256"],
        ),
        "linear_selected_step_616": (
            JointTemporalScorer(),
            linear_best_path,
            linear["best_checkpoint_sha256"],
        ),
        "gru_step_5632": (
            OrderedCrossModalGRUScorer(),
            gru_final_path,
            gru["history"][-1]["checkpoint_sha256"],
        ),
    }
    logits_by_model: dict[str, np.ndarray] = {}
    for name, (model, checkpoint_path, checkpoint_hash) in model_specs.items():
        _load_checkpoint(model, checkpoint_path, checkpoint_hash)
        logits_by_model[name] = _predict(model, dev)

    model_metrics = {
        name: grouped_metrics(logits, targets, categories)
        for name, logits in logits_by_model.items()
    }
    comparisons = {
        "gru_5632_minus_linear_5632": paired_group_bootstrap(
            logits_by_model["gru_step_5632"],
            logits_by_model["linear_step_5632"],
            targets,
            groups,
            draws=draws,
        ),
        "linear_selected_616_minus_linear_5632": paired_group_bootstrap(
            logits_by_model["linear_selected_step_616"],
            logits_by_model["linear_step_5632"],
            targets,
            groups,
            draws=draws,
        ),
    }

    output.mkdir(parents=True, exist_ok=False)
    prediction_path = output / "predictions.jsonl"
    with prediction_path.open("x", encoding="utf-8", newline="\n") as stream:
        for index, row in enumerate(development_rows):
            prediction = {
                name: {
                    "logits": logits[index].tolist(),
                    "probabilities": torch.softmax(torch.from_numpy(logits[index]), dim=0).tolist(),
                }
                for name, logits in logits_by_model.items()
            }
            stream.write(
                json.dumps(
                    {
                        "sample_id": row["sample_id"],
                        "group_id": str(row["group_id"]),
                        "category": str(row["category"]),
                        "target": int(targets[index]),
                        "predictions": prediction,
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
    report = {
        "status": "FRESH_GROUP_DEVELOPMENT_EVALUATION_COMPLETE_NO_TRAINING_NO_CHECKPOINT_SELECTION",
        "task": "four-choice video-to-IMU timestamp correspondence",
        "input_count": len(development_rows),
        "unique_group_count": len(np.unique(groups)),
        "category_group_counts": {
            category: int(len(set(groups[categories == category])))
            for category in sorted(set(categories.tolist()))
        },
        "category_sample_counts": {
            category: int((categories == category).sum())
            for category in sorted(set(categories.tolist()))
        },
        "metrics": model_metrics,
        "paired_group_bootstrap": comparisons,
        "models": {
            name: {
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_hash,
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
            }
            for name, (model, checkpoint_path, checkpoint_hash) in model_specs.items()
        },
        "plan_sha256": sha256_file(plan_path),
        "aggregate_report_sha256": sha256_file(corpus / "aggregate-report.json"),
        "corpus_sha256": aggregate["sha256"],
        "predictions_sha256": sha256_file(prediction_path),
        "sealed_audit_accessed": False,
        "training_performed": False,
        "checkpoint_selected_from_fresh_development": False,
        "interpretation": (
            "one-time development generalization diagnostic; group-level only, "
            "participant independence unknown; not a sealed audit or product task"
        ),
    }
    (output / "results.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--linear-run", type=Path, required=True)
    parser.add_argument("--gru-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=2_000)
    args = parser.parse_args()
    result = evaluate(
        plan_path=args.plan,
        corpus=args.corpus,
        linear_run=args.linear_run,
        gru_run=args.gru_run,
        output=args.output,
        draws=args.draws,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "input_count": result["input_count"],
                "unique_group_count": result["unique_group_count"],
                "metrics": {
                    name: value["overall"] for name, value in result["metrics"].items()
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
