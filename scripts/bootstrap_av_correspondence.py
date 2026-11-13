"""Quartet-cluster percentile intervals on saved development predictions."""

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.audit_correspondence_validation import sha, write
from tiny_omni_decision.decision_math import (
    brier_score,
    expected_calibration_error,
    negative_log_likelihood,
)


def bootstrap(records, conditions, draws=10000, seed=17):
    groups = {}
    for index, row in enumerate(records):
        groups.setdefault(row["quartet_id"], []).append(index)
    if not groups or any(len(g) != 4 for g in groups.values()):
        raise ValueError("complete quartets required")
    indices = np.array(list(groups.values()))
    rng = np.random.default_rng(seed)
    samples = indices[rng.integers(0, len(indices), (draws, len(indices)))].reshape(draws, -1)
    metrics = {}
    for name, condition in conditions.items():
        p = np.asarray(condition["probabilities"], dtype=np.float64)
        y = np.asarray(condition["targets"])
        expected = [r["target_index"] for r in records]
        if name == "reversed_options":
            expected = [3 - target for target in expected]
        if p.shape != (len(records), 4) or y.tolist() != expected:
            raise ValueError("prediction alignment mismatch")
        if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(1), 1):
            raise ValueError("invalid probabilities")
        values = []
        for selected in samples:
            probabilities, targets = p[selected].tolist(), y[selected].tolist()
            values.append(
                [
                    float((p[selected].argmax(1) == y[selected]).mean()),
                    negative_log_likelihood(probabilities, targets),
                    brier_score(probabilities, targets),
                    expected_calibration_error(probabilities, targets),
                ]
            )
        metrics[name] = np.asarray(values)
    names = ("accuracy", "nll", "brier", "ece")

    def intervals(values):
        return dict(zip(names, np.quantile(values, [0.025, 0.975], axis=0).T.tolist(), strict=True))

    return dict(
        draws=draws,
        seed=seed,
        groups=len(groups),
        unit="quartet",
        method="95% percentile; shared cluster resamples across conditions",
        limitations=(
            "development validation selected checkpoint; not blind; participant overlap possible"
        ),
        conditions={n: intervals(v) for n, v in metrics.items()},
        paired_best_minus_untrained=intervals(metrics["best"] - metrics["untrained"]),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("output already exists")
    digest = sha(args.predictions)
    data = json.loads(args.predictions.read_text(encoding="utf-8"))
    result = bootstrap(data["records"], data["conditions"])
    if sha(args.predictions) != digest:
        raise ValueError("predictions changed")
    result.update(predictions_sha256=digest, script_sha256=sha(Path(__file__)))
    write(args.output, result)


if __name__ == "__main__":
    main()
