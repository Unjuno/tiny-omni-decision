"""Paired-state uncertainty of the fixed soft replay endpoint; saved predictions only."""

import hashlib
import json
from pathlib import Path

import numpy as np


def compare(old, new):
    assert [(r["id"], r["options"], r["target"]) for r in old] == [
        (r["id"], r["options"], r["target"]) for r in new
    ]
    groups = sorted(
        {r["id"].split(":")[1] if r["context"] == "original" else r["group"] for r in old}
    )
    lookup = {g: i for i, g in enumerate(groups)}
    stats = np.zeros((len(groups), 4))
    for a, b in zip(old, new, strict=True):
        y = a["options"].index(a["target"])
        p, q = np.asarray(a["probabilities"]), np.asarray(b["probabilities"])
        group = a["id"].split(":")[1] if a["context"] == "original" else a["group"]
        stats[lookup[group]] += [
            int(q.argmax() == y) - int(p.argmax() == y),
            np.log(max(p[y], 1e-12)) - np.log(max(q[y], 1e-12)),
            (q * q).sum() - 2 * q[y] - (p * p).sum() + 2 * p[y],
            1,
        ]
    rng = np.random.default_rng(17)
    draws = np.asarray(
        [stats[rng.integers(len(groups), size=len(groups))].sum(0) for _ in range(10000)]
    )
    point = stats.sum(0)
    return {
        "states": len(groups),
        "questions": len(old),
        **{
            k: {
                "delta": float(point[i] / point[3]),
                "ci95": np.quantile(draws[:, i] / draws[:, 3], [0.025, 0.975]).tolist(),
            }
            for i, k in enumerate(["accuracy", "nll", "brier"])
        },
    }


def main():
    root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    paths = {
        "parent": root / "context-replay-context-run-01-all-encoder/predictions-0.json",
        "hard": root / "context-replay-context-run-01-all-encoder/predictions-384.json",
        "soft": root / "context-soft-replay-context-run-01-all-encoder/predictions-384.json",
    }
    data = {k: json.loads(p.read_text()) for k, p in paths.items()}
    result = {
        "method": (
            "paired underlying-state bootstrap; 10000 draws; seed17; percentile95%; "
            "unadjusted; sample weighted; NLL floor1e-12; observed development"
        ),
        "hashes": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths.values()},
        "contrasts": {},
    }
    for baseline in ("parent", "hard"):
        result["contrasts"][baseline] = {}
        for kind in ("original", "plain", "irrelevant", "necessary"):
            result["contrasts"][baseline][kind] = compare(
                [r for r in data[baseline] if r["context"] == kind],
                [r for r in data["soft"] if r["context"] == kind],
            )
    Path("docs/pretrained_reuse/R2_SOFT_REPLAY_UNCERTAINTY_20261013.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result["contrasts"], indent=2))


if __name__ == "__main__":
    main()
