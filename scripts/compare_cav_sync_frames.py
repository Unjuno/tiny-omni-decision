"""Paired quartet bootstrap of fixed one/eight-frame development predictions."""

import json
from pathlib import Path

import numpy as np

root = Path("C:/CodexArtifacts/pretrained-reuse-r1")
one = root / "cav-mae-sync-correspondence-valid-01"
eight = root / "cav-mae-sync-correspondence-eight-01"
reports = [json.loads((p / "result.json").read_text()) for p in (one, eight)]
assert reports[0]["routing_sha256"] == reports[1]["routing_sha256"]
assert [r["id"] for r in reports[0]["records"]] == [r["id"] for r in reports[1]["records"]]
predictions = [np.load(p / "predictions.npz") for p in (one, eight)]
assert np.array_equal(predictions[0]["targets"], predictions[1]["targets"])
routes = json.loads(
    Path(
        "C:/CodexArtifacts/pretrained-reuse-r2/"
        "perception-correspondence-valid-audit-01/valid-routing.json"
    ).read_text()
)
groups = {}
for i, row in enumerate(routes):
    groups.setdefault(row["quartet_id"], []).append(i)
assert len(groups) == 50 and all(len(g) == 4 for g in groups.values())
clusters = np.array(list(groups.values()))
draws = clusters[np.random.default_rng(17).integers(0, 50, (10000, 50))].reshape(10000, 200)
values = []
for data in predictions:
    p, y = data["probabilities"], data["targets"]
    assert y.tolist() == [r["target_index"] for r in routes]
    values.append(
        np.stack(
            [
                (p.argmax(1) == y).astype(float),
                -np.log(p[np.arange(len(y)), y]),
                ((p - np.eye(4)[y]) ** 2).sum(1),
            ],
            axis=1,
        )
    )
delta = values[1] - values[0]
report = {
    "comparison": "eight minus one frames; observed development validation",
    "seed": 17,
    "draws": 10000,
    "clusters": 50,
    "method": "paired quartet percentile 95%; unadjusted exploratory intervals",
    "delta": dict(zip(["accuracy", "nll", "brier"], delta.mean(0).tolist(), strict=True)),
    "intervals": dict(
        zip(
            ["accuracy", "nll", "brier"],
            np.quantile(delta[draws].mean(1), [0.025, 0.975], axis=0).T.tolist(),
            strict=True,
        )
    ),
}
path = eight / "comparison.json"
with path.open("x", encoding="utf-8") as f:
    json.dump(report, f, indent=2)
print(json.dumps(report, indent=2))
