"""Verify and compare BERT vs Ruri on identical grouped JCommonsenseQA predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _metrics(targets: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predictions = probabilities.argmax(axis=-1)
    one_hot = np.eye(probabilities.shape[-1], dtype=np.float64)[targets]
    return {
        "accuracy": float((predictions == targets).mean()),
        "nll": float(
            -np.log(np.maximum(probabilities[np.arange(len(targets)), targets], 1e-12)).mean()
        ),
        "brier": float(((probabilities - one_hot) ** 2).sum(axis=-1).mean()),
    }


def verify_checkpoint_reload(run_dir: Path, predictions: list[dict[str, Any]]) -> dict[str, Any]:
    cache = torch.load(run_dir / "frozen-features.pt", map_location="cpu", weights_only=True)
    feature_size = int(cache["validation_features"].shape[-1])
    scorer = nn.Sequential(
        nn.Linear(feature_size, 128), nn.Tanh(), nn.Dropout(0.1), nn.Linear(128, 1)
    )
    checkpoint = run_dir / "best-readout.pt"
    scorer.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    scorer.eval()
    with torch.inference_mode():
        probabilities = (
            scorer(cache["validation_features"].reshape(-1, feature_size))
            .reshape(-1, 5)
            .softmax(-1)
            .numpy()
        )
    saved = np.asarray([row["probabilities"] for row in predictions])
    if probabilities.shape != saved.shape:
        raise ValueError("reloaded checkpoint and saved predictions have different shapes")
    delta = float(np.max(np.abs(probabilities - saved)))
    if delta != 0.0:
        raise ValueError(f"checkpoint reload mismatch: max probability delta {delta}")
    return {
        "status": "pass",
        "rows": int(saved.shape[0]),
        "max_probability_abs_delta": delta,
        "checkpoint_sha256": sha256_file(checkpoint),
    }


def paired_group_bootstrap(
    candidate: list[dict[str, Any]],
    baseline: list[dict[str, Any]],
    *,
    iterations: int = 10_000,
    seed: int = 17,
) -> dict[str, Any]:
    for left, right in zip(candidate, baseline, strict=True):
        for key in ("sample_id", "content_group_id", "target_index"):
            if left[key] != right[key]:
                raise ValueError(f"prediction mismatch for {key}: {left[key]} != {right[key]}")
    targets = np.asarray([row["target_index"] for row in candidate], dtype=np.int64)
    candidate_probs = np.asarray([row["probabilities"] for row in candidate], dtype=np.float64)
    baseline_probs = np.asarray([row["probabilities"] for row in baseline], dtype=np.float64)
    group_indices: dict[str, list[int]] = {}
    for index, row in enumerate(candidate):
        group_indices.setdefault(row["content_group_id"], []).append(index)
    groups = list(group_indices.values())
    rng = random.Random(seed)
    differences = {name: [] for name in ("accuracy", "nll", "brier")}
    for _ in range(iterations):
        selected = [rng.randrange(len(groups)) for _ in groups]
        indices = np.fromiter(
            (index for group in selected for index in groups[group]), dtype=np.int64
        )
        candidate_metrics = _metrics(targets[indices], candidate_probs[indices])
        baseline_metrics = _metrics(targets[indices], baseline_probs[indices])
        for name in differences:
            differences[name].append(candidate_metrics[name] - baseline_metrics[name])
    candidate_point = _metrics(targets, candidate_probs)
    baseline_point = _metrics(targets, baseline_probs)
    return {
        "rows": len(candidate),
        "groups": len(groups),
        "bootstrap": {
            "method": "paired question-group bootstrap; sample groups with replacement",
            "iterations": iterations,
            "seed": seed,
            "interval": "percentile 95%",
            "candidate_minus_baseline": {
                name: {
                    "point_difference": candidate_point[name] - baseline_point[name],
                    "bootstrap_95_percentile_interval": [
                        float(np.quantile(values, 0.025)),
                        float(np.quantile(values, 0.975)),
                    ],
                }
                for name, values in differences.items()
            },
        },
        "candidate": candidate_point,
        "baseline": baseline_point,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    candidate = load_jsonl(args.candidate_run / "validation-predictions.jsonl")
    baseline = load_jsonl(args.baseline_run / "validation-predictions.jsonl")
    if len(candidate) != len(baseline):
        raise ValueError("validation prediction counts differ")
    reload_result = verify_checkpoint_reload(args.candidate_run, candidate)
    comparison = paired_group_bootstrap(
        candidate, baseline, iterations=args.iterations, seed=args.seed
    )
    candidate_report = json.loads((args.candidate_run / "run-report.json").read_text())
    baseline_report = json.loads((args.baseline_run / "run-report.json").read_text())
    if candidate_report["data"]["validation_sha256"] != baseline_report["validation_sha256"]:
        raise ValueError("runs did not use the same validation rows")
    result = {
        "status": "paired_development_validation_only",
        "candidate_model": candidate_report["model"],
        "baseline_model": {
            "id": baseline_report["model_id"],
            "revision": baseline_report["model_revision"],
            "frozen_parameters": baseline_report["encoder_parameters_frozen"],
        },
        "same_validation_sha256": candidate_report["data"]["validation_sha256"],
        "same_ids_targets_groups": True,
        "reload_verification": reload_result,
        **comparison,
        "uses_validation_for_checkpoint_selection_and_comparison": True,
        "official_test_or_sealed_audit_opened": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
