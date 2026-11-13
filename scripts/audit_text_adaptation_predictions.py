"""Posthoc paired audit of existing predictions; no inference or tuning."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = [args.root / "text" / f"{split}.jsonl" for split in ("train", "validation")]
    paths += [
        args.root / name / "predictions.json"
        for name in ("text-crossencoder-zero-shot-01", "text-crossencoder-adapt-v1")
    ]
    train, validation = [
        [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()] for p in paths[:2]
    ]
    instructions = {
        d.get("instructions", key)
        for row in train
        for key, d in json.loads(row["questions"]).items()
    }
    meta = {}
    for row in validation:
        for key, definition in json.loads(row["questions"]).items():
            meta[f"n4ze3m/typed-decisions-synth:{row['state_id']}:{key}"] = (
                row["state_id"],
                definition["type"],
                definition.get("instructions", key) in instructions,
            )
    predictions = [json.loads(p.read_text(encoding="utf-8")) for p in paths[2:]]
    ids = [p["id"] for p in predictions[0]]
    assert len(ids) == len(set(ids)) == 2540
    assert ids == [p["id"] for p in predictions[1]]
    assert set(ids) == set(meta)
    arrays = []
    for rows in predictions:
        values = []
        for row in rows:
            p = np.asarray(row["probabilities"], dtype=float)
            assert np.isfinite(p).all() and (p >= 0).all()
            assert abs(p.sum() - 1) < 1e-5
            target = row["options"].index(row["target"])
            y = np.eye(len(p))[target]
            values.append(
                [
                    float(p.argmax() == target),
                    -np.log(max(p[target], 1e-30)),
                    float(((p - y) ** 2).sum()),
                ]
            )
        arrays.append(np.array(values))
    assert all(
        a["options"] == b["options"] and a["target"] == b["target"]
        for a, b in zip(*predictions, strict=True)
    )
    groups = {"all": np.ones(len(ids), dtype=bool)}
    for kind in ("choice", "noul", "score"):
        groups[kind] = np.array([meta[i][1] == kind for i in ids])
    for seen in (True, False):
        groups["seen_exact_instruction" if seen else "unseen_exact_instruction"] = np.array(
            [meta[i][2] == seen for i in ids]
        )

    def metrics(a):
        return dict(zip(("accuracy", "nll", "brier"), a.mean(axis=0).tolist(), strict=True))

    strata = {
        name: {
            "count": int(mask.sum()),
            "zero_shot": metrics(arrays[0][mask]),
            "adapted": metrics(arrays[1][mask]),
        }
        for name, mask in groups.items()
    }
    states = sorted({meta[i][0] for i in ids})
    differences = arrays[1] - arrays[0]
    sums = np.array(
        [differences[[meta[i][0] == state for i in ids]].sum(axis=0) for state in states]
    )
    counts = np.array([sum(meta[i][0] == state for i in ids) for state in states])
    rng = np.random.default_rng(17)
    samples = []
    for _ in range(10000):
        chosen = rng.integers(len(states), size=len(states))
        samples.append(sums[chosen].sum(axis=0) / counts[chosen].sum())
    intervals = np.quantile(samples, [0.025, 0.975], axis=0)
    result = {
        "scope": "posthoc observed-validation diagnosis; no new blind evaluation",
        "hashes": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "strata": strata,
        "paired_state_bootstrap": {
            "states": len(states),
            "seed": 17,
            "replicates": 10000,
            "method": (
                "percentile 95%; resample states with replacement; "
                "question-weighted metrics; adapted minus zero-shot"
            ),
            "difference": metrics(differences),
            "intervals": dict(zip(("accuracy", "nll", "brier"), intervals.T.tolist(), strict=True)),
        },
        "limitations": [
            "Selection on this validation is not accounted for by intervals.",
            "Exact wording novelty is not semantic/task novelty.",
            "No inference, new training, sealed data or calibration fit.",
        ],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"strata": strata, "bootstrap": result["paired_state_bootstrap"]}, indent=2))


if __name__ == "__main__":
    main()
