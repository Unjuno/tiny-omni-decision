"""Cluster-bootstrap paired AV source-candidate control predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

CONTROL_NAMES = (
    "zero_audio",
    "shuffled_audio",
    "zero_video",
    "reverse_temporal_delta",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_predictions(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def validate_pair(reference: list[dict], comparison: list[dict], name: str) -> None:
    if len(reference) != len(comparison):
        raise ValueError(f"paired control has different record count: {name}")
    for first, second in zip(reference, comparison, strict=True):
        for key in (
            "record_id",
            "video_id",
            "candidate_ids",
            "target_candidate_ids",
            "none_visible",
        ):
            if first[key] != second[key]:
                raise ValueError(f"paired control changed {key} for {name}")


def prediction_arrays(rows: list[dict]) -> dict[str, np.ndarray]:
    if not rows:
        raise ValueError("cannot bootstrap an empty prediction set")
    probabilities = []
    targets = []
    for row in rows:
        ids = row["candidate_ids"]
        target_ids = set(row["target_candidate_ids"])
        p = [float(row["candidate_probabilities"][candidate_id]) for candidate_id in ids]
        p.append(float(row["none_visible_probability"]))
        y = [candidate_id in target_ids for candidate_id in ids]
        y.append(bool(row["none_visible"]))
        positive_count = sum(y)
        if positive_count == 0:
            raise ValueError(f"record has no positive target: {row['record_id']}")
        probabilities.append(p)
        targets.append(np.asarray(y, dtype=np.float64) / positive_count)
    return {
        "video_ids": np.asarray([row["video_id"] for row in rows]),
        "accuracy": np.asarray([row["top1_correct"] for row in rows], dtype=np.float64),
        "nll": np.asarray([row["nll"] for row in rows], dtype=np.float64),
        "confidence": np.asarray([row["top1_confidence"] for row in rows], dtype=np.float64),
        "brier": np.asarray(
            [
                np.square(np.asarray(p, dtype=np.float64) - y).sum()
                for p, y in zip(probabilities, targets, strict=True)
            ],
            dtype=np.float64,
        ),
    }


def ece(confidence: np.ndarray, accuracy: np.ndarray, bins: int = 10) -> float:
    result = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        mask = (confidence >= lower) & (
            confidence < upper if index < bins - 1 else confidence <= upper
        )
        if np.any(mask):
            result += float(mask.mean()) * abs(
                float(confidence[mask].mean() - accuracy[mask].mean())
            )
    return result


def point_metrics(arrays: dict[str, np.ndarray], indices: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(arrays["accuracy"][indices].mean()),
        "nll": float(arrays["nll"][indices].mean()),
        "brier": float(arrays["brier"][indices].mean()),
        "ece_10_bins": ece(arrays["confidence"][indices], arrays["accuracy"][indices]),
    }


def paired_group_bootstrap(
    reference: list[dict],
    comparison: list[dict],
    *,
    resamples: int = 10_000,
    seed: int = 17,
) -> dict:
    validate_pair(reference, comparison, "paired-bootstrap")
    first = prediction_arrays(reference)
    second = prediction_arrays(comparison)
    video_ids = first["video_ids"]
    groups = sorted(set(video_ids.tolist()))
    group_rows = {group: np.flatnonzero(video_ids == group) for group in groups}
    point_a = point_metrics(first, np.arange(len(reference)))
    point_b = point_metrics(second, np.arange(len(comparison)))
    deltas = {metric: [] for metric in point_a}
    rng = np.random.default_rng(seed)
    for _ in range(resamples):
        sampled_groups = rng.choice(groups, size=len(groups), replace=True)
        indices = np.concatenate([group_rows[group] for group in sampled_groups])
        sampled_a = point_metrics(first, indices)
        sampled_b = point_metrics(second, indices)
        for metric in deltas:
            deltas[metric].append(sampled_a[metric] - sampled_b[metric])
    summary = {}
    for metric in deltas:
        values = np.asarray(deltas[metric], dtype=np.float64)
        summary[metric] = {
            "reference_minus_comparison": point_a[metric] - point_b[metric],
            "lower_95_percentile": float(np.quantile(values, 0.025)),
            "upper_95_percentile": float(np.quantile(values, 0.975)),
            "bootstrap_probability_reference_better": float(
                np.mean(values > 0) if metric == "accuracy" else np.mean(values < 0)
            ),
        }
    return {
        "records": len(reference),
        "independent_video_groups": len(groups),
        "resamples": resamples,
        "seed": seed,
        "interval": "two-sided 95% percentile interval from paired video-cluster bootstrap",
        "metrics": summary,
    }


def run(prediction_dir: Path, output: Path, *, resamples: int, seed: int) -> dict:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite bootstrap result: {output}")
    baseline = read_predictions(prediction_dir / "predictions-joint.jsonl")
    offset_baseline = read_predictions(prediction_dir / "predictions-joint_offset_eligible.jsonl")
    controls = {}
    for name in CONTROL_NAMES:
        comparison = read_predictions(prediction_dir / f"predictions-{name}.jsonl")
        controls[f"joint_vs_{name}"] = paired_group_bootstrap(
            baseline,
            comparison,
            resamples=resamples,
            seed=seed,
        )
    offset = read_predictions(prediction_dir / "predictions-offset_audio.jsonl")
    controls["source_audio_vs_offset_audio_same_subset"] = paired_group_bootstrap(
        offset_baseline,
        offset,
        resamples=resamples,
        seed=seed,
    )
    prediction_hashes = {
        path.name: sha256_file(path) for path in sorted(prediction_dir.glob("predictions-*.jsonl"))
    }
    result = {
        "status": "PASS_PAIRED_VIDEO_CLUSTER_BOOTSTRAP",
        "scope": (
            "exploratory fixed-checkpoint controls on the same non-blind development "
            "partition used for checkpoint selection"
        ),
        "input_prediction_sha256": prediction_hashes,
        "resamples": resamples,
        "seed": seed,
        "comparison_direction": (
            "all metrics are reference minus comparison; positive Accuracy and "
            "negative NLL/Brier/ECE favor the reference"
        ),
        "controls": controls,
        "limits": [
            "80 development videos are below the roadmap's provisional 200-scene floor.",
            "Checkpoint selection and paired controls use the same non-blind development "
            "partition; intervals are not blind confirmatory evidence.",
            "Perception Test source-candidate ranking uses oracle event intervals and "
            "object tracks.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.resamples < 1:
        parser.error("--resamples must be positive")
    print(
        json.dumps(
            run(args.prediction_dir, args.output, resamples=args.resamples, seed=args.seed),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
