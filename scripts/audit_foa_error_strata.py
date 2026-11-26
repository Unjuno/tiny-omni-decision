"""Train-only fixed strata for preserved FOA predictions; no algorithm fitting."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts.range_generalization import sha, write

p = Path("C:/CodexArtifacts/pretrained-reuse-r2/starss23-foa-direction-probe-01/predictions.jsonl")
assert sha(p) == "c6ae866a84f1c6dc80eb78044cd660d402b59583f9eb780785758875109c0841"
rows = [json.loads(s) for s in p.read_text().splitlines()]
groups = defaultdict(list)
for r in rows:
    distance = r["distance_cm"]
    group = (
        "0-100cm"
        if distance <= 100
        else "101-200cm"
        if distance <= 200
        else "201-400cm"
        if distance <= 400
        else "over400cm"
    )
    groups["distance:" + group].append(r)
    groups["raw_class:" + str(r["raw_class"])].append(r)
result = {
    "source_sha256": sha(p),
    "scope": (
        "post-hoc train-only; adjacent ticks correlated; fixed distance bins, "
        "no exclusions, no parameter fitting"
    ),
    "strata": {},
}
for key, items in groups.items():
    error = np.array([r["error_deg"] for r in items])
    result["strata"][key] = {
        "ticks": len(items),
        "recordings": len({r["recording"] for r in items}),
        "mean_deg": float(error.mean()),
        "median_deg": float(np.median(error)),
        "within20": float((error <= 20).mean()),
        "within45": float((error <= 45).mean()),
    }
write(Path("docs/pretrained_reuse/R2_FOA_ERROR_STRATA_20261013.json"), result)
print(json.dumps({k: v for k, v in result["strata"].items() if k.startswith("distance")}, indent=2))
