"""Paired observer-cluster comparison for the fixed iNaturalist readout diagnostic."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.fit_native_feature_inat_diagnostic import BASELINE_DIR, FEATURE_DIR, sha256

ROOT = Path(
    r"C:\CodexArtifacts\pretrained-reuse-r2\native-feature-decision-inat-20261031-attempt-01"
)
PREDICTIONS = ROOT / "validation-predictions.jsonl"
BASELINE_PREDICTIONS = BASELINE_DIR / "validation-predictions.jsonl"
FEATURE_ROWS = FEATURE_DIR / "feature-rows.jsonl"


def metrics(probabilities: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    accuracy = float(np.mean(predictions == targets))
    per_class = [
        float(np.mean(predictions[targets == index] == targets[targets == index]))
        for index in np.unique(targets)
    ]
    onehot = np.eye(probabilities.shape[1], dtype=np.float64)[targets]
    true_probabilities = probabilities[np.arange(len(targets)), targets]
    ece = 0.0
    for low in np.linspace(0.0, 1.0, 16)[:-1]:
        high = low + 1.0 / 15.0
        mask = (confidence > low) & (confidence <= high)
        if mask.any():
            ece += float(mask.mean()) * abs(
                float(confidence[mask].mean()) - float(np.mean(predictions[mask] == targets[mask]))
            )
    return {
        "accuracy": accuracy,
        "macro_accuracy": float(np.mean(per_class)),
        "nll": float(-np.log(np.clip(true_probabilities, 1e-12, 1.0)).mean()),
        "brier": float(((probabilities - onehot) ** 2).sum(axis=1).mean()),
        "ece_15_bins": ece,
        "mean_confidence": float(confidence.mean()),
    }


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main() -> None:
    classes = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))["option_schema"][
        "taxon_id"
    ]
    candidate = read_jsonl(PREDICTIONS)
    baseline = read_jsonl(BASELINE_PREDICTIONS)
    rows = [row for row in read_jsonl(FEATURE_ROWS) if row["split"] == "validation"]
    if not (len(candidate) == len(baseline) == len(rows) == 430):
        raise ValueError("fixed validation coverage changed")
    for new, old, source in zip(candidate, baseline, rows, strict=True):
        if (new["sample_id"], old["sample_id"], source["sample_id"]) != (new["sample_id"],) * 3:
            raise ValueError("prediction identity/order mismatch")
        if (new["target_taxon_id"], str(old["target_taxon_id"]), str(source["taxon_id"])) != (
            new["target_taxon_id"],
        ) * 3:
            raise ValueError("target mismatch")
        if new["observer_group_sha256"] != source["observer_group_sha256"]:
            raise ValueError("observer-group attribution mismatch")

    targets = np.asarray([classes.index(row["target_taxon_id"]) for row in candidate])
    new_p = np.asarray(
        [[row["probabilities_by_taxon_id"][c] for c in classes] for row in candidate]
    )
    old_p = np.asarray([[row["probabilities_by_taxon_id"][c] for c in classes] for row in baseline])
    if not (np.isfinite(new_p).all() and np.isfinite(old_p).all()):
        raise ValueError("nonfinite probability")
    if not np.allclose(new_p.sum(axis=1), 1.0, atol=1e-6) or not np.allclose(
        old_p.sum(axis=1), 1.0, atol=1e-6
    ):
        raise ValueError("probabilities do not normalize")

    point = {"baseline": metrics(old_p, targets), "native_feature": metrics(new_p, targets)}
    point["native_feature_minus_baseline"] = {
        key: point["native_feature"][key] - point["baseline"][key] for key in point["baseline"]
    }
    groups = np.asarray([row["observer_group_sha256"] for row in candidate])
    unique_groups = sorted(set(groups.tolist()))
    group_to_indices = {group: np.flatnonzero(groups == group) for group in unique_groups}
    rng = np.random.default_rng(17)
    draws = {key: [] for key in point["native_feature_minus_baseline"]}
    for _ in range(10_000):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([group_to_indices[group] for group in sampled])
        before, after = (
            metrics(old_p[indices], targets[indices]),
            metrics(new_p[indices], targets[indices]),
        )
        for key in draws:
            draws[key].append(after[key] - before[key])
    intervals = {
        key: [float(value) for value in np.quantile(values, [0.025, 0.975])]
        for key, values in draws.items()
    }
    report = {
        "status": "PASS_PAIRED_OBSERVER_CLUSTER_DIAGNOSTIC",
        "n": len(candidate),
        "observer_groups": len(unique_groups),
        "bootstrap": {
            "replicates": 10_000,
            "seed": 17,
            "resampling_unit": "observer group",
            "interval": "percentile 95%",
        },
        "point_metrics": point,
        "paired_difference_intervals": intervals,
        "inputs_sha256": {
            "candidate_predictions": sha256(PREDICTIONS),
            "baseline_predictions": sha256(BASELINE_PREDICTIONS),
            "feature_rows": sha256(FEATURE_ROWS),
        },
        "interpretation_limit": (
            "same post-selection development validation; intervals are descriptive and not "
            "blind or task-generalization guarantees"
        ),
    }
    path = ROOT / "comparison.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
