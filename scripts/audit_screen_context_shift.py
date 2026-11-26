"""Read-only normalization and linear-logit contribution audit."""

import json
from pathlib import Path

import numpy as np
import torch

from scripts.range_generalization import sha, write

root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
base = root / "groundcua-classifier-01"
context = root / "groundcua-fullscreen-features-01"
run = root / "groundcua-context-comparison-01"
rows = json.loads((base / "records.json").read_text())
screens = json.loads((context / "records.json").read_text())
lookup = {x["screen"]: i for i, x in enumerate(screens)}
x = np.concatenate(
    [
        np.load(base / "features.npy"),
        np.load(context / "features.npy")[[lookup[r["screen"]] for r in rows]],
    ],
    1,
)
n = np.load(run / "context/normalization.npz")
z = (x - n["mean"]) / n["std"]
train = np.array([r["split"] == "train" for r in rows])
result = {
    "features": {},
    "logits": {},
    "hashes": {
        str(p): sha(p)
        for p in [
            base / "features.npy",
            base / "records.json",
            context / "features.npy",
            context / "records.json",
            run / "context/normalization.npz",
            run / "context/initial.pt",
            run / "context/final.pt",
        ]
    },
}
for name, sl in [("crop", slice(0, 768)), ("screen", slice(768, None))]:
    result["features"][name] = {}
    for split, mask in [("train", train), ("validation", ~train)]:
        v = z[mask, sl]
        result["features"][name][split] = {
            "rms": float(np.sqrt((v * v).mean())),
            "abs_p99": float(np.quantile(abs(v), 0.99)),
            "abs_max": float(abs(v).max()),
            "fraction_abs_gt5": float((abs(v) > 5).mean()),
        }
for checkpoint in ["initial", "final"]:
    w = torch.load(run / f"context/{checkpoint}.pt", weights_only=True)["weight"].numpy()
    result["logits"][checkpoint] = {}
    for split, mask in [("train", train), ("validation", ~train)]:
        blocks = {}
        for name, sl in [("crop", slice(0, 768)), ("screen", slice(768, None))]:
            v = z[mask, sl] @ w[:, sl].T
            v = v - v.mean(1, keepdims=True)
            blocks[name] = float(np.sqrt((v * v).mean()))
        result["logits"][checkpoint][split] = blocks
write(Path("docs/pretrained_reuse/R2_SCREEN_CONTEXT_SHIFT_20261013.json"), result)
print(json.dumps({"features": result["features"], "logits": result["logits"]}, indent=2))
