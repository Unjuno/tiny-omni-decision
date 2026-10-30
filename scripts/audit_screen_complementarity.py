"""Saved-prediction complementarity, never a deployable oracle score."""

import json
from pathlib import Path

import numpy as np

from scripts.range_generalization import sha, write
from scripts.train_screen_pilot import CATEGORIES

root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
clip = root / "tinyclip-screen-zero-shot-01"
dino = root / "groundcua-expanded-classifier-01"
rows = json.loads((clip / "rows.json").read_text())
reference = [
    r
    for r in json.loads((root / "groundcua-classifier-01/records.json").read_text())
    if r["split"] == "validation"
]
assert rows == reference
result = json.loads((dino / "result.json").read_text())
dpath = dino / f"epoch-{result['best_epoch']}.npy"
p = np.load(clip / "probabilities.npy")
q = np.load(dpath)
y = np.array([r["target"] for r in rows])
assert p.shape == q.shape == (len(y), 8)
a, b = p.argmax(1), q.argmax(1)
report = {
    "count": len(y),
    "categories": CATEGORIES,
    "clip_predicted_counts": np.bincount(a, minlength=8).tolist(),
    "target_counts": np.bincount(y, minlength=8).tolist(),
    "clip_confusion": np.bincount(y * 8 + a, minlength=64).reshape(8, 8).tolist(),
    "dino_confusion": np.bincount(y * 8 + b, minlength=64).reshape(8, 8).tolist(),
    "both_correct": int(((a == y) & (b == y)).sum()),
    "clip_only_correct": int(((a == y) & (b != y)).sum()),
    "dino_only_correct": int(((a != y) & (b == y)).sum()),
    "both_wrong": int(((a != y) & (b != y)).sum()),
    "oracle_top1_selector_accuracy": float(((a == y) | (b == y)).mean()),
    "scope": "oracle uses labels; not an ensemble evaluation or model capacity ceiling",
    "hashes": {str(f): sha(f) for f in [clip / "rows.json", clip / "probabilities.npy", dpath]},
}
write(Path("docs/pretrained_reuse/R2_SCREEN_COMPLEMENTARITY_20261013.json"), report)
print({k: v for k, v in report.items() if k not in ["hashes", "clip_confusion", "dino_confusion"]})
