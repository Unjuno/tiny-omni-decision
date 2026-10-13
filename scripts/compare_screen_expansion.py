"""Paired app-cluster uncertainty from saved screen predictions, no fitting."""

import json
from pathlib import Path

import numpy as np

from scripts.range_generalization import sha, write

root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
old = root / "groundcua-classifier-01"
new = root / "groundcua-expanded-classifier-01"
a = json.loads((old / "result.json").read_text())
b = json.loads((new / "result.json").read_text())
rows = [r for r in json.loads((old / "records.json").read_text()) if r["split"] == "validation"]
p = np.load(old / f"epoch-{a['best_epoch']}.npy")
q = np.load(new / f"epoch-{b['best_epoch']}.npy")
y = np.array([r["target"] for r in rows])
apps = sorted({r["app"] for r in rows})
assert p.shape == q.shape == (723, 8)
d = np.column_stack(
    [
        (q.argmax(1) == y).astype(int) - (p.argmax(1) == y).astype(int),
        np.log(np.maximum(p[np.arange(len(y)), y], 1e-30))
        - np.log(np.maximum(q[np.arange(len(y)), y], 1e-30)),
        np.ones(len(y)),
    ]
)
stats = np.array([d[np.array([r["app"] == app for r in rows])].sum(0) for app in apps])
rng = np.random.default_rng(17)
draws = np.array([stats[rng.integers(4, size=4)].sum(0) for _ in range(10000)])
result = {
    "method": (
        "four-app paired bootstrap,10000 draws,seed17,percentile95%,"
        "sample-weighted,unadjusted; very low cluster count; selected development checkpoints"
    ),
    "accuracy_delta": float(d[:, 0].mean()),
    "accuracy_ci": np.quantile(draws[:, 0] / draws[:, 2], [0.025, 0.975]).tolist(),
    "nll_delta": float(d[:, 1].mean()),
    "nll_ci": np.quantile(draws[:, 1] / draws[:, 2], [0.025, 0.975]).tolist(),
    "apps": {app: {"old": a["apps"][app], "expanded": b["apps"][app]} for app in apps},
    "expanded_prior": b["prior"],
    "hashes": {
        str(f): sha(f)
        for f in [
            old / "records.json",
            old / f"epoch-{a['best_epoch']}.npy",
            new / f"epoch-{b['best_epoch']}.npy",
        ]
    },
}
write(Path("docs/pretrained_reuse/R2_SCREEN_EXPANSION_UNCERTAINTY_20261013.json"), result)
print({k: v for k, v in result.items() if k not in ["apps", "expanded_prior", "hashes"]})
