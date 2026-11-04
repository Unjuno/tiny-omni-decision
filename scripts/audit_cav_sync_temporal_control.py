"""Post-hoc temporal dependence check on fixed saved AV features."""

import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.train_sensor_subjects import metrics

root = Path("C:/CodexArtifacts/pretrained-reuse-r1/cav-mae-sync-correspondence-eight-01")
out = root.parent / "cav-mae-sync-temporal-control-01"
out.mkdir(exist_ok=False)
r = json.loads((root / "result.json").read_text())
f = root / "features.npz"
assert hashlib.sha256(f.read_bytes()).hexdigest() == r["feature_sha256"]
features = np.load(f)
a = features["audio"]
v = features["video"]
a = a / np.linalg.norm(a, axis=-1, keepdims=True)
v = v / np.linalg.norm(v, axis=-1, keepdims=True)
p = Path(
    "C:/CodexArtifacts/pretrained-reuse-r2/perception-correspondence-valid-audit-01/valid-routing.json"
)
assert hashlib.sha256(p.read_bytes()).hexdigest() == r["routing_sha256"]
routes = json.loads(p.read_text())
index = {x["id"]: i for i, x in enumerate(r["records"])}
y = np.array([x["target_index"] for x in routes])
conditions = {}
pred = {}
for name, audio in [("matched", a), ("reversed_audio_time", a[:, ::-1])]:
    logits = np.array(
        [
            [
                (audio[index[x["audio_video_id"]]] * v[index[c]]).sum(-1).mean()
                for c in x["candidate_video_ids"]
            ]
            for x in routes
        ]
    )
    prob = np.exp(logits - logits.max(1, keepdims=True))
    prob /= prob.sum(1, keepdims=True)
    conditions[name] = metrics(prob, y)
    pred[name] = prob
assert np.array_equal(pred["matched"], np.load(root / "predictions.npz")["probabilities"])
groups = {}
for i, x in enumerate(routes):
    groups.setdefault(x["quartet_id"], []).append(i)
g = np.array(list(groups.values()))
assert g.shape == (50, 4)
samples = g[np.random.default_rng(17).integers(0, 50, (10000, 50))].reshape(10000, 200)
x = pred["matched"]
z = pred["reversed_audio_time"]
d = np.stack(
    [
        (z.argmax(1) == y).astype(float) - (x.argmax(1) == y),
        -np.log(z[np.arange(200), y]) + np.log(x[np.arange(200), y]),
    ],
    1,
)
report = dict(
    conditions=conditions,
    delta_reversed_minus_matched=d.mean(0).tolist(),
    accuracy_nll_interval95=np.quantile(d[samples].mean(1), [0.025, 0.975], axis=0).T.tolist(),
    seed=17,
    draws=10000,
    unit="quartet",
    feature_sha256=r["feature_sha256"],
    interpretation="Post-hoc observed development; feature order reversal, not waveform reversal",
)
np.savez(out / "predictions.npz", **pred, targets=y)
(out / "result.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
