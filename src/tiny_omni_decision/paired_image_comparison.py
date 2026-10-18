"""Paired image bootstrap for additive accuracy/NLL/Brier differences."""

import numpy as np


def paired_image_metrics(first, second, *, replicates=10000, seed=17):
    if not first or len(first) != len(second) or replicates < 1:
        raise ValueError("nonempty paired records and positive replicates required")
    groups, seen = {}, set()
    for a, b in zip(first, second, strict=True):
        if any(a[k] != b[k] for k in ("id", "image_id", "target", "options")) or a["id"] in seen:
            raise ValueError("record identity mismatch or duplicate")
        seen.add(a["id"])
        values = []
        for row in (a, b):
            p = np.asarray(row["probabilities"], dtype=np.float64)
            if (
                p.shape != (len(row["options"]),)
                or not np.isfinite(p).all()
                or p.min() < 0
                or abs(p.sum() - 1) > 1e-6
            ):
                raise ValueError("invalid probabilities")
            target = row["options"].index(row["target"])
            truth = np.zeros(len(p))
            truth[target] = 1
            values.append(
                np.array(
                    [
                        float(p.argmax() == target),
                        -np.log(max(p[target], 1e-12)),
                        ((p - truth) ** 2).sum(),
                    ]
                )
            )
        groups.setdefault(a["image_id"], []).append(values[1] - values[0])
    if len({len(v) for v in groups.values()}) != 1:
        raise ValueError("balanced question counts per image required")
    delta = np.array([np.mean(v, axis=0) for v in groups.values()])
    indices = np.random.default_rng(seed).integers(len(delta), size=(replicates, len(delta)))
    result = {}
    for index, name in enumerate(("accuracy", "nll", "brier")):
        draws = delta[:, index][indices].mean(1)
        result[name] = dict(
            difference=float(delta[:, index].mean()),
            percentile_95_interval=np.quantile(draws, [0.025, 0.975]).tolist(),
        )
    return dict(
        groups=len(groups),
        replicates=replicates,
        seed=seed,
        direction="second minus first",
        quantiles="linear percentile",
        metrics=result,
    )
