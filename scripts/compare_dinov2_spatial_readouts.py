"""Compare paired global and spatial readout predictions by image-cluster bootstrap."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))
sys.path.insert(0, str(REPOSITORY / "src"))

from tiny_omni_decision.dataset import CLEVR4_TAXONOMIES  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def prediction_arrays(path: Path) -> tuple[list[str], dict[str, np.ndarray]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    tasks = list(CLEVR4_TAXONOMIES)
    by_key = {(row["image_id"], row["taxonomy"]): row for row in rows}
    images = sorted({row["image_id"] for row in rows})
    if len(by_key) != len(rows) or len(rows) != len(images) * len(tasks):
        raise ValueError("prediction rows are not one row per image and taxonomy")
    metrics = {
        name: np.empty((len(images), len(tasks)), dtype=np.float64)
        for name in ("accuracy", "nll", "brier")
    }
    for image_index, image_id in enumerate(images):
        for task_index, task in enumerate(tasks):
            row = by_key.get((image_id, task))
            if row is None:
                raise ValueError("missing paired image/taxonomy prediction")
            options = CLEVR4_TAXONOMIES[task]
            probs = np.asarray(row["probabilities"], dtype=np.float64)
            target = options.index(row["target"])
            if probs.shape != (len(options),) or not np.isfinite(probs).all():
                raise ValueError("invalid probability vector")
            if abs(float(probs.sum()) - 1.0) > 1e-5 or (probs < 0).any():
                raise ValueError("probabilities are not a distribution")
            predicted = int(probs.argmax())
            if row["prediction"] != options[predicted]:
                raise ValueError("saved prediction does not match probabilities")
            one_hot = np.zeros(len(options), dtype=np.float64)
            one_hot[target] = 1.0
            metrics["accuracy"][image_index, task_index] = float(predicted == target)
            metrics["nll"][image_index, task_index] = -np.log(max(probs[target], 1e-12))
            metrics["brier"][image_index, task_index] = np.square(probs - one_hot).sum()
    return images, metrics


def paired_image_bootstrap(
    global_metrics: dict[str, np.ndarray],
    spatial_metrics: dict[str, np.ndarray],
    *,
    repeats: int = 5000,
    seed: int = 17,
) -> dict:
    if repeats < 100 or set(global_metrics) != set(spatial_metrics):
        raise ValueError("invalid bootstrap input")
    shape = {value.shape for value in (*global_metrics.values(), *spatial_metrics.values())}
    if len(shape) != 1 or len(next(iter(shape))) != 2:
        raise ValueError("paired metric arrays must have equal [images,tasks] shape")
    rng = np.random.default_rng(seed)
    result = {
        "seed": seed,
        "repeats": repeats,
        "resampling_unit": "image_id",
        "interval": "percentile 95%",
    }
    for metric in sorted(global_metrics):
        delta = spatial_metrics[metric] - global_metrics[metric]
        macro_delta = delta.mean(axis=1)
        task_names = list(CLEVR4_TAXONOMIES)
        metrics = {
            "macro": macro_delta,
            **{task: delta[:, index] for index, task in enumerate(task_names)},
        }
        result[metric] = {}
        for task, values in metrics.items():
            boot = np.empty(repeats, dtype=np.float64)
            for repeat in range(repeats):
                sampled = rng.integers(0, len(values), size=len(values))
                boot[repeat] = values[sampled].mean()
            result[metric][task] = {
                "point_delta": float(values.mean()),
                "ci95": [float(x) for x in np.quantile(boot, [0.025, 0.975])],
                "bootstrap_fraction_delta_le_zero": float(np.mean(boot <= 0)),
            }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5000)
    args = parser.parse_args()
    global_path = args.run / "global" / "best-predictions-reloaded.json"
    spatial_path = args.run / "spatial" / "best-predictions-reloaded.json"
    image_ids, global_metrics = prediction_arrays(global_path)
    spatial_ids, spatial_metrics = prediction_arrays(spatial_path)
    if image_ids != spatial_ids:
        raise ValueError("global/spatial validation IDs or order differ")
    result = {
        "run": str(args.run),
        "global_predictions_sha256": sha256(global_path),
        "spatial_predictions_sha256": sha256(spatial_path),
        "images": len(image_ids),
        "taxonomies": list(CLEVR4_TAXONOMIES),
        "bootstrap": paired_image_bootstrap(global_metrics, spatial_metrics, repeats=args.repeats),
        "interpretation_limit": (
            "development validation was previously used for model selection; "
            "intervals are descriptive, not blind-test inference"
        ),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
