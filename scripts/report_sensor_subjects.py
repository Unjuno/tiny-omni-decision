"""Read-only post-run diagnostics; no training, audit or checkpoint selection."""

import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.range_generalization import sha, write
from scripts.train_sensor_subjects import ROOT, metrics


def main():
    run = ROOT / "sensor-subject-training-01"
    output = ROOT / "sensor-subject-diagnostics-01"
    output.mkdir(exist_ok=False)
    summary = json.loads((run / "result.json").read_text())
    manifest = json.loads((ROOT / "harth-subject-windows-01/result.json").read_text())
    hashes = {"train": set(), "validation": set()}
    for row in manifest["subjects"]:
        if sha(Path(row["path"])) != row["sha256"]:
            raise ValueError("input hash mismatch")
        with np.load(row["path"], allow_pickle=False) as data:
            for x in data["values"]:
                # Exact physical values, without timestamps/labels/subject identifiers.
                hashes[row["split"]].add(hashlib.sha256(x.astype("<f4").tobytes()).hexdigest())
    overlap = len(hashes["train"] & hashes["validation"])
    result = {"exact_value_overlap": overlap, "models": {}}
    predictions = {}
    for mode in ("temporal", "static"):
        epoch = summary[mode]["best_epoch"]
        with np.load(run / mode / f"epoch-{epoch}.npz", allow_pickle=False) as data:
            p, y, subjects, mixed, ids = [data[k] for k in (
                "probabilities", "targets", "subjects", "mixed", "ids"
            )]
        predictions[mode] = (p, y, subjects, ids)
        result["models"][mode] = {
            "subjects": {s: metrics(p[subjects == s], y[subjects == s])
                         for s in sorted(set(subjects.tolist()))},
            "mixed": metrics(p[mixed], y[mixed]),
            "unmixed": metrics(p[~mixed], y[~mixed]),
            "confusion": np.bincount(y * 12 + p.argmax(1), minlength=144).reshape(12, 12).tolist(),
        }
    a, y, subjects, ids = predictions["temporal"]
    b, by, bs, bid = predictions["static"]
    if not all(np.array_equal(x, z) for x, z in ((y, by), (subjects, bs), (ids, bid))):
        raise ValueError("paired identity mismatch")
    # Exhaustive resampling of three subject clusters: 3^3=27 equally likely draws.
    groups = sorted(set(subjects.tolist()))
    differences = []
    for selected in np.ndindex(*(len(groups),) * len(groups)):
        indices = np.concatenate([np.flatnonzero(subjects == groups[i]) for i in selected])
        am, bm = metrics(a[indices], y[indices]), metrics(b[indices], y[indices])
        differences.append([am[k] - bm[k] for k in ("accuracy", "nll", "brier")])
    result["paired_temporal_minus_static"] = {
        "method": "exhaustive subject-cluster bootstrap; 27 draws; percentile linear quantiles",
        "seed": None, "independent_subjects": len(groups),
        "warning": "Only 3 subjects; highly discrete development interval after selection",
        "interval_95": dict(zip(("accuracy", "nll", "brier"),
                                np.quantile(differences, [0.025, 0.975], axis=0).T.tolist(),
                                strict=True)),
    }
    write(output / "result.json", result)
    print(json.dumps({"overlap": overlap, "paired": result["paired_temporal_minus_static"],
                      "subsets": {m: {s: {k: v[k] for k in ("accuracy", "nll", "count")}
                                      for s, v in d.items() if s in ("mixed", "unmixed")}
                                  for m, d in result["models"].items()}}, indent=2))


if __name__ == "__main__":
    main()
