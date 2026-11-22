"""Group-bootstrap uncertainty for the fixed TRAIN-only absent-target audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from scripts.pretrained_reuse.audit_spoken_palette_absent_target import binary_auc


def summarize(
    records: list[dict[str, Any]], *, replicates: int = 10_000, seed: int = 17
) -> dict[str, Any]:
    """Resample speaker groups while preserving each source's three paired conditions."""
    if replicates < 100:
        raise ValueError("at least 100 bootstrap replicates are required")
    by_group: dict[str, dict[str, dict[str, dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for row in records:
        group = str(row.get("speaker_group", ""))
        sample = str(row.get("sample_id", ""))
        condition = str(row.get("condition", ""))
        valid_conditions = {"matched", "target_absent", "distractor_absent"}
        if not group or not sample or condition not in valid_conditions:
            raise ValueError("prediction row has an invalid group, sample, or condition")
        if condition in by_group[group][sample]:
            raise ValueError("duplicate sample/condition prediction row")
        by_group[group][sample][condition] = row
    if len(by_group) < 2:
        raise ValueError("group bootstrap requires at least two independent groups")
    if any(
        set(conditions) != {"matched", "target_absent", "distractor_absent"}
        for samples in by_group.values()
        for conditions in samples.values()
    ):
        raise ValueError("every source must have all three paired conditions")

    group_ids = sorted(by_group)
    rng = np.random.default_rng(seed)

    def metrics(selected_groups: list[str]) -> dict[str, float]:
        deltas_absent: list[float] = []
        deltas_distractor: list[float] = []
        accuracy_matched: list[float] = []
        accuracy_distractor: list[float] = []
        positives: list[float] = []
        negatives: list[float] = []
        for group in selected_groups:
            for conditions in by_group[group].values():
                matched = conditions["matched"]
                absent = conditions["target_absent"]
                distractor = conditions["distractor_absent"]
                deltas_absent.append(
                    float(absent["max_probability"]) - float(matched["max_probability"])
                )
                deltas_distractor.append(
                    float(distractor["max_probability"]) - float(matched["max_probability"])
                )
                accuracy_matched.append(float(bool(matched["prediction_matches_target_position"])))
                accuracy_distractor.append(
                    float(bool(distractor["prediction_matches_target_position"]))
                )
                positives.append(float(absent["uncertainty_score"]))
                negatives.extend(
                    (
                        float(matched["uncertainty_score"]),
                        float(distractor["uncertainty_score"]),
                    )
                )
        return {
            "target_absent_max_probability_delta": float(np.mean(deltas_absent)),
            "distractor_absent_max_probability_delta": float(np.mean(deltas_distractor)),
            "matched_target_present_accuracy": float(np.mean(accuracy_matched)),
            "distractor_absent_target_present_accuracy": float(np.mean(accuracy_distractor)),
            "target_absence_uncertainty_auc": binary_auc(
                [1] * len(positives) + [0] * len(negatives), positives + negatives
            ),
        }

    point = metrics(group_ids)
    draws = {name: [] for name in point}
    for _ in range(replicates):
        selected = rng.choice(group_ids, size=len(group_ids), replace=True).tolist()
        draw = metrics(selected)
        for name, value in draw.items():
            draws[name].append(value)
    return {
        "method": "speaker-group paired bootstrap; resample whole speaker groups with replacement",
        "seed": seed,
        "replicates": replicates,
        "independent_groups": len(group_ids),
        "source_pairs": sum(len(samples) for samples in by_group.values()),
        "percentile_interval": [0.025, 0.975],
        "metrics": {
            name: {
                "estimate": estimate,
                "ci95": [
                    float(np.quantile(draws[name], 0.025)),
                    float(np.quantile(draws[name], 0.975)),
                ],
            }
            for name, estimate in point.items()
        },
        "interpretation_limit": (
            "Fixed-checkpoint TRAIN-only diagnostic; intervals describe these 24 speaker groups, "
            "not generalization or a blind evaluation."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.output.exists() or Path(f"{args.output}.sha256").exists():
        raise FileExistsError("refusing to overwrite bootstrap output")
    payload = args.predictions.read_bytes()
    records = [json.loads(line) for line in payload.decode("utf-8").splitlines() if line.strip()]
    result = summarize(records, replicates=args.replicates, seed=args.seed)
    result["predictions_sha256"] = hashlib.sha256(payload).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(
        "utf-8"
    )
    with args.output.open("xb") as stream:
        stream.write(encoded)
    digest = hashlib.sha256(encoded).hexdigest()
    with Path(f"{args.output}.sha256").open("x", encoding="ascii") as stream:
        stream.write(digest + "\n")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
