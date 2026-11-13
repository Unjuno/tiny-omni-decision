"""Paired state-cluster diagnostics for completed numeric embedding predictions."""

import json
from collections import Counter
from pathlib import Path

import numpy as np

from scripts.range_generalization import sha, write


def compare(left, right, seed=17, draws=10000):
    if len(left) != len(right) or len({r["id"] for r in left}) != len(left):
        raise ValueError("duplicate or mismatched predictions")
    groups = {}
    transitions = Counter()
    for a, b in zip(left, right, strict=True):
        if any(a[k] != b[k] for k in ["id", "target", "options"]):
            raise ValueError("identity mismatch")
        pa, pb = np.array(a["probabilities"]), np.array(b["probabilities"])
        for p in [pa, pb]:
            if (
                len(p) != len(a["options"])
                or not np.isfinite(p).all()
                or (p < 0).any()
                or abs(p.sum() - 1) > 1e-5
            ):
                raise ValueError("invalid probabilities")
        target = a["options"].index(a["target"])
        ca, cb = int(pa.argmax() == target), int(pb.argmax() == target)
        transitions[f"{ca}_to_{cb}"] += 1
        truth = np.eye(len(pa))[target]
        difference = np.array(
            [
                cb - ca,
                -np.log(max(pb[target], 1e-12)) + np.log(max(pa[target], 1e-12)),
                ((pb - truth) ** 2).sum() - ((pa - truth) ** 2).sum(),
            ]
        )
        group = a["id"].rsplit(":", 1)[0]
        groups.setdefault(group, []).append(difference)
    sums = np.array([np.sum(v, axis=0) for v in groups.values()])
    counts = np.array([len(v) for v in groups.values()])
    indices = np.random.default_rng(seed).integers(len(groups), size=(draws, len(groups)))
    boot = sums[indices].sum(1) / counts[indices].sum(1)[:, None]
    point = sums.sum(0) / counts.sum()
    return dict(
        questions=len(left),
        states=len(groups),
        group_sizes=dict(Counter(counts.tolist())),
        transitions=dict(transitions),
        seed=seed,
        draws=draws,
        metrics={
            name: dict(
                difference=float(point[i]),
                percentile_95_interval=np.quantile(boot[:, i], [0.025, 0.975]).tolist(),
            )
            for i, name in enumerate(["accuracy", "nll", "brier"])
        },
    )


def main():
    root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    parent = root / "conditional-repair-comparison-vocabulary-01"
    run = root / "numeric-embedding-training-01"
    out = root / "numeric-embedding-paired-01"
    if out.exists():
        raise FileExistsError(out)
    results = {}
    hashes = {}
    for step in [0, 128, 256, 384, 512]:
        for source in ["original", "curriculum"]:
            paths = [p / f"predictions-{step}-{source}.json" for p in [parent, run]]
            values = [json.loads(p.read_text()) for p in paths]
            results[f"{step}-{source}"] = compare(*values)
            hashes.update({str(p): sha(p) for p in paths})
    out.mkdir()
    report = dict(
        comparisons=results,
        hashes=hashes,
        limitation="Post-hoc descriptive development diagnostics; no multiplicity correction.",
        inference=False,
        training=False,
    )
    write(out / "result.json", report)
    write(Path("docs/pretrained_reuse/R2_NUMERIC_EMBEDDING_PAIRED_20261012.json"), report)
    print(json.dumps({k: v for k, v in results.items() if k.startswith("512")}, indent=2))


if __name__ == "__main__":
    main()
