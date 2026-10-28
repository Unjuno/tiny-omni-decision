"""Quantify paired DROID later-shard readout differences from frozen predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

OPTIONS = ("stop", "x-", "x+", "y-", "y+", "z-", "z+")
ARMS = ("sensor", "video_sensor")
METRICS = ("accuracy", "nll", "brier")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_rows(rows: list[dict[str, Any]]) -> None:
    if len(rows) < 2:
        raise ValueError("at least two prediction rows are required")
    ids = [row.get("id") for row in rows]
    groups = [row.get("group_id") for row in rows]
    if any(not value for value in ids + groups) or len(set(ids)) != len(ids):
        raise ValueError("prediction IDs must be present and unique")
    if len(set(groups)) != len(groups):
        raise ValueError("each row must represent a unique independent episode group")
    for row in rows:
        if row.get("site") is None:
            raise ValueError("every row must have a collection site")
        target = row.get("target_option_id")
        if target not in OPTIONS:
            raise ValueError(f"unknown target option: {target!r}")
        probabilities = row.get("probabilities")
        if not isinstance(probabilities, dict):
            raise ValueError("each row must contain arm probability vectors")
        for arm in ARMS:
            values = np.asarray(probabilities.get(arm), dtype=np.float64)
            if (
                values.shape != (len(OPTIONS),)
                or not np.isfinite(values).all()
                or (values < 0).any()
                or not np.isclose(values.sum(), 1.0, atol=1e-8)
            ):
                raise ValueError(f"invalid {arm} probabilities for {row['id']}")


def _metrics(rows: list[dict[str, Any]], arm: str) -> dict[str, float]:
    targets = np.asarray([OPTIONS.index(row["target_option_id"]) for row in rows])
    probabilities = np.asarray(
        [row["probabilities"][arm] for row in rows], dtype=np.float64
    )
    correct = probabilities.argmax(axis=1) == targets
    target_probabilities = probabilities[np.arange(len(rows)), targets]
    one_hot = np.eye(len(OPTIONS), dtype=np.float64)[targets]
    return {
        "accuracy": float(correct.mean()),
        "nll": float(-np.log(np.clip(target_probabilities, 1e-15, 1.0)).mean()),
        "brier": float(((probabilities - one_hot) ** 2).sum(axis=1).mean()),
    }


def paired_site_cluster_bootstrap(
    rows: list[dict[str, Any]], *, repetitions: int = 10_000, seed: int = 17
) -> dict[str, Any]:
    """Bootstrap paired model deltas by resampling whole collection sites."""
    _validate_rows(rows)
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    sites = np.asarray([str(row["site"]) for row in rows])
    unique_sites = np.unique(sites)
    if len(unique_sites) < 2:
        raise ValueError("at least two collection sites are required")
    rows_by_site = {
        site: [row for row in rows if str(row["site"]) == site]
        for site in unique_sites
    }

    point = {
        metric: _metrics(rows, "video_sensor")[metric] - _metrics(rows, "sensor")[metric]
        for metric in METRICS
    }
    rng = np.random.default_rng(seed)
    draws = {metric: np.empty(repetitions, dtype=np.float64) for metric in METRICS}
    for index in range(repetitions):
        sampled_sites = rng.choice(unique_sites, size=len(unique_sites), replace=True)
        sample = [row for site in sampled_sites for row in rows_by_site[site]]
        joint = _metrics(sample, "video_sensor")
        sensor = _metrics(sample, "sensor")
        for metric in METRICS:
            draws[metric][index] = joint[metric] - sensor[metric]

    return {
        "unit": "collection_site",
        "site_count": int(len(unique_sites)),
        "repetitions": repetitions,
        "seed": seed,
        "resampling": "sample sites with replacement and retain every row in each sampled site",
        "point_delta_video_sensor_minus_sensor": point,
        "percentile_95_ci": {
            metric: [
                float(np.quantile(draws[metric], 0.025)),
                float(np.quantile(draws[metric], 0.975)),
            ]
            for metric in METRICS
        },
    }


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--predictions-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    if not args.predictions.is_file():
        raise FileNotFoundError(args.predictions)
    observed_hash = sha256_file(args.predictions)
    if observed_hash.lower() != args.predictions_sha256.lower():
        raise ValueError("prediction file SHA-256 does not match the expected value")
    rows = [
        json.loads(line)
        for line in args.predictions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    _validate_rows(rows)
    arms = {arm: _metrics(rows, arm) for arm in ARMS}
    bootstrap = paired_site_cluster_bootstrap(
        rows, repetitions=args.repetitions, seed=args.seed
    )
    result = {
        "status": "PASS_POSTHOC_PAIRED_SITE_BOOTSTRAP",
        "scope": "fixed prediction file only; no model, feature, or dataset reload",
        "prediction_file_sha256": observed_hash,
        "count": len(rows),
        "unique_episode_groups": len({row["group_id"] for row in rows}),
        "unique_sites": len({row["site"] for row in rows}),
        "fixed_prediction_metrics": arms,
        "paired_bootstrap": bootstrap,
        "claim_limit": (
            "post-hoc uncertainty diagnostic on an already observed official-train "
            "development sample; not blind or product evidence"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    summary = {
        key: result[key]
        for key in ("status", "count", "unique_sites", "paired_bootstrap")
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
