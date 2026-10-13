import numpy as np

"""Read-only exact-crop label ambiguity audit; no training or audit split access."""

import collections
import hashlib
import json
from pathlib import Path

root = Path("C:/CodexArtifacts/pretrained-reuse-r2/groundcua-classifier-01")
p = root / "records.json"
rows = json.loads(p.read_text())
result = {"source_sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "splits": {}}
for split in sorted({r["split"] for r in rows}):
    selected = [r for r in rows if r["split"] == split]
    groups = collections.defaultdict(list)
    for r in selected:
        groups[r["crop_sha256"]].append(r)
    conflicts = [g for g in groups.values() if len({r["target"] for r in g}) > 1]
    unavoidable = sum(
        len(g) - max(collections.Counter(r["target"] for r in g).values()) for g in groups.values()
    )
    result["splits"][split] = {
        "rows": len(selected),
        "unique_crops": len(groups),
        "conflicting_crops": len(conflicts),
        "conflicting_rows": sum(map(len, conflicts)),
        "minimum_errors_crop_only": unavoidable,
        "empirical_deterministic_crop_ceiling": 1 - unavoidable / len(selected),
        "conflicts": [
            {
                "crop_sha256": g[0]["crop_sha256"],
                "labels": dict(collections.Counter(r["target"] for r in g)),
                "screens": sorted({r["screen"] for r in g}),
            }
            for g in conflicts
        ],
    }
Path("docs/pretrained_reuse/R2_SCREEN_CROP_AMBIGUITY_20261013.json").write_text(
    json.dumps(result, indent=2) + "\n"
)
print(
    json.dumps(
        {k: {a: b for a, b in v.items() if a != "conflicts"} for k, v in result["splits"].items()},
        indent=2,
    )
)

# A single fixed cosine nearest-neighbor control; no k/temperature selection.

features = np.load(root / "features.npy", allow_pickle=False)
train = np.array([r["split"] == "train" for r in rows])
y = np.array([r["target"] for r in rows])
features = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
nearest = (features[~train] @ features[train].T).argmax(1)
predicted, target = y[train][nearest], y[~train]
result["cosine_one_neighbor"] = {
    "feature_sha256": hashlib.sha256((root / "features.npy").read_bytes()).hexdigest(),
    "accuracy": float((predicted == target).mean()),
    "balanced_accuracy": float(
        np.mean([(predicted[target == c] == c).mean() for c in np.unique(target)])
    ),
    "train_indices": np.flatnonzero(train)[nearest].tolist(),
    "predictions": predicted.tolist(),
    "note": "hard retrieval diagnostic; no calibrated probabilities or NLL claimed",
}
Path("docs/pretrained_reuse/R2_SCREEN_CROP_AMBIGUITY_20261013.json").write_text(
    json.dumps(result, indent=2) + "\n"
)
