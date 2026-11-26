"""Compare completed controlled development runs; never read sealed evaluation."""

import argparse
import hashlib
import json
import math
from pathlib import Path

CONTROLLED = (
    "seed",
    "steps",
    "batch",
    "lr",
    "weight_decay",
    "clip",
    "dropout",
    "optimizer_reset",
    "initialization",
    "selector",
    "paths",
    "train_ids",
    "validation_ids",
)


def paired_accuracy(rows, left, right):
    """Return per-state paired accuracy differences, preserving all questions."""
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids) or any([r["id"] for r in x] != ids for x in (left, right)):
        raise ValueError("duplicate or mismatched question IDs/order")
    groups = {}
    for row, a, b in zip(rows, left, right, strict=True):
        for prediction in (a, b):
            if prediction["target"] != row["target"] or prediction["options"] != row["options"]:
                raise ValueError("target/options mismatch")
            probabilities = prediction["probabilities"]
            if (
                len(probabilities) != len(row["options"])
                or any(not math.isfinite(v) or v < 0 or v > 1 for v in probabilities)
                or abs(sum(probabilities) - 1) > 1e-5
            ):
                raise ValueError("invalid probabilities")

        def correct(prediction, row=row):
            index = max(range(len(row["options"])), key=prediction["probabilities"].__getitem__)
            return int(row["options"][index] == row["target"])

        groups.setdefault(row["group"], []).append(correct(b) - correct(a))
    if not groups or len({len(v) for v in groups.values()}) != 1:
        raise ValueError("expected equal-size state groups")
    return [sum(v) / len(v) for v in groups.values()]


def main():
    import numpy as np

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "candidate", "curriculum", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    def load(root, name):
        return json.loads((root / name).read_text(encoding="utf-8"))

    roots = (args.baseline, args.candidate)
    configs = [load(p, "config.json") for p in roots]
    for key in CONTROLLED:
        if configs[0][key] != configs[1][key]:
            raise ValueError(f"controlled field mismatch: {key}")
    corpus_hash = hashlib.sha256(args.curriculum.read_bytes()).hexdigest()
    expected = [
        value
        for path, value in configs[0]["paths"].items()
        if Path(path).resolve() == args.curriculum.resolve()
    ]
    if expected != [corpus_hash]:
        raise ValueError("curriculum input must match the frozen run corpus hash")
    results = [load(p, "result.json") for p in roots]
    if any(r["status"] != "PASS" or r["steps"] != 2048 for r in results):
        raise ValueError("both full runs must have completed")
    histories = [load(p, "history.json") for p in roots]
    if any([r["step"] for r in h] != list(range(0, 2049, 128)) for h in histories):
        raise ValueError("incomplete evaluation history")
    rows = [json.loads(x) for x in args.curriculum.read_text(encoding="utf-8").splitlines()]
    comparisons = {}
    for label, steps in [(str(s), (s, s)) for s in (512, 1024, 1536, 1920, 2048)] + [
        ("selected", tuple(r["best_step"] for r in results))
    ]:
        predictions = [
            load(p, f"predictions-{s}-curriculum.json") for p, s in zip(roots, steps, strict=True)
        ]
        differences = np.array(paired_accuracy(rows, *predictions))
        rng = np.random.default_rng(20261012)
        draws = differences[rng.integers(0, len(differences), (10000, len(differences)))].mean(1)
        comparisons[label] = {
            "steps": steps,
            "candidate_minus_baseline_accuracy": float(differences.mean()),
            "paired_state_percentile_95_interval": np.quantile(draws, [0.025, 0.975]).tolist(),
            "metrics": [
                next(r for r in h if r["step"] == s) for h, s in zip(histories, steps, strict=True)
            ],
        }
    report = {
        "roots": list(map(str, roots)),
        "results": results,
        "comparisons": comparisons,
        "bootstrap_seed": 20261012,
        "replicates": 10000,
        "limitation": "Observed development validation, also used for selection; not blind audit.",
        "config_sha256": [
            hashlib.sha256((p / "config.json").read_bytes()).hexdigest() for p in roots
        ],
        "curriculum_sha256": corpus_hash,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
