from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CORPUS = Path("C:/CodexArtifacts/tqpp/fullclip-v1/validation.jsonl").resolve()
EVALUATION_ROOT = Path("C:/CodexArtifacts/tqpp/evaluations").resolve()
OUTPUT = Path("C:/CodexArtifacts/tqpp/physionpp-reference-comparison.json").resolve()
BOOTSTRAP_SEED = 17
BOOTSTRAP_REPLICATES = 10_000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_predictions(name: str) -> tuple[dict[str, object], list[dict[str, object]]]:
    directory = EVALUATION_ROOT / name
    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    predictions = [
        json.loads(line)
        for line in (directory / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    return metrics, predictions


def ece(probabilities: np.ndarray, targets: np.ndarray, bins: int = 15) -> float:
    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    correct = predicted == targets
    result = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        selected = (confidence >= lower) & (
            (confidence < upper) if index < bins - 1 else (confidence <= upper)
        )
        if selected.any():
            result += float(selected.mean()) * abs(
                float(correct[selected].mean()) - float(confidence[selected].mean())
            )
    return result


def metric_values(
    predictions: list[dict[str, object]], indices: np.ndarray
) -> dict[str, float]:
    targets = np.asarray([int(row["target"]) for row in predictions], dtype=np.int64)[indices]
    probs = np.asarray([row["option_probabilities"] for row in predictions], dtype=np.float64)[
        indices
    ]
    chosen = probs[np.arange(len(targets)), targets]
    one_hot = np.eye(probs.shape[1], dtype=np.float64)[targets]
    return {
        "accuracy": float((probs.argmax(axis=1) == targets).mean()),
        "nll": float(-np.log(np.clip(chosen, 1e-12, 1.0)).mean()),
        "brier": float(np.square(probs - one_hot).sum(axis=1).mean()),
        "ece": ece(probs, targets),
    }


def main() -> None:
    validation_hash = sha256_file(CORPUS)
    v1_metrics, v1 = load_predictions("teacher-v1-reference")
    e_metrics, e_long = load_predictions("teacher-v2-e-long-selected-step1536")
    if validation_hash != v1_metrics["validation_sha256"]:
        raise ValueError("Teacher v1 evaluation used another validation corpus")
    if validation_hash != e_metrics["validation_sha256"]:
        raise ValueError("E-long evaluation used another validation corpus")
    if v1_metrics["sample_id_order_sha256"] != e_metrics["sample_id_order_sha256"]:
        raise ValueError("reference predictions use different sample orders")
    if len(v1) != len(e_long) or not v1:
        raise ValueError("reference prediction lengths differ or are empty")
    for left, right in zip(v1, e_long, strict=True):
        for field in ("sample_id", "target", "option_labels", "source_asset_group_id"):
            if left[field] != right[field]:
                raise ValueError(f"paired reference mismatch for {field}: {left['sample_id']}")

    groups: dict[str, list[int]] = {}
    for index, row in enumerate(v1):
        groups.setdefault(str(row["source_asset_group_id"]), []).append(index)
    group_indices = list(groups.values())
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    differences: dict[str, list[float]] = {
        name: [] for name in ("accuracy", "nll", "brier", "ece")
    }
    for _ in range(BOOTSTRAP_REPLICATES):
        selected_groups = rng.integers(0, len(group_indices), size=len(group_indices))
        indices = np.asarray(
            [item for group_index in selected_groups for item in group_indices[group_index]],
            dtype=np.int64,
        )
        before = metric_values(v1, indices)
        after = metric_values(e_long, indices)
        for name in differences:
            differences[name].append(after[name] - before[name])

    intervals = {
        name: {
            "point_delta": float(
                metric_values(e_long, np.arange(len(e_long)))[name]
                - metric_values(v1, np.arange(len(v1)))[name]
            ),
            "percentile_95_interval": [
                float(np.quantile(samples, 0.025)),
                float(np.quantile(samples, 0.975)),
            ],
        }
        for name, samples in differences.items()
    }
    result = {
        "status": "paired development-set comparison; not a blind audit",
        "validation_sha256": validation_hash,
        "sample_id_order_sha256": v1_metrics["sample_id_order_sha256"],
        "examples": len(v1),
        "scene_seed_clusters": len(groups),
        "bootstrap": {
            "unit": "source_asset_group_id (scenario:trial_seed)",
            "seed": BOOTSTRAP_SEED,
            "replicates": BOOTSTRAP_REPLICATES,
            "interval": "percentile 95% over paired E-long minus v1 deltas",
        },
        "teacher_v1": {
            "adapter_sha256": v1_metrics["adapter_sha256"],
            "metrics": v1_metrics["modality:video"],
        },
        "teacher_v2_e_long_selected_step1536": {
            "adapter_sha256": e_metrics["adapter_sha256"],
            "metrics": e_metrics["modality:video"],
        },
        "e_long_minus_v1": intervals,
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
