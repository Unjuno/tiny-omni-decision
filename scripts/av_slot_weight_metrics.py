"""Fixed paired speaker intervals and prospective readout eligibility gates."""

import numpy as np

from scripts.lombard_correspondence import choice_metrics


def eligible(candidate, baseline, strata):
    return bool(
        candidate["accuracy"] >= baseline["accuracy"]
        and candidate["nll"] <= baseline["nll"] * 0.99
        and candidate["brier"] <= baseline["brier"]
        and candidate["ece"] <= baseline["ece"] + 0.005
        and all(s["count"] < 100 or s["accuracy_delta"] >= -0.02 for s in strata)
    )


def paired_comparison(candidate, candidate_logp, baseline, baseline_logp, targets, groups):
    y = np.asarray(targets)
    metrics = [
        choice_metrics(p, y, logp)
        for p, logp in [(baseline, baseline_logp), (candidate, candidate_logp)]
    ]
    if len(groups) != len(y) or any(not isinstance(g, str) or not g for g in groups):
        raise ValueError("complete independent speaker groups required")
    labels = sorted(set(groups))
    stats = []
    for p, logp in [(baseline, baseline_logp), (candidate, candidate_logp)]:
        good = p.argmax(1) == y
        conf = p.max(1)
        loss = -logp[np.arange(len(y)), y]
        b = p.copy()
        b[np.arange(len(y)), y] -= 1
        b = np.square(b).sum(1)
        bins = np.minimum((conf * 15).astype(int), 14)
        rows = []
        for g in labels:
            mask = np.asarray(groups) == g
            rows.append(
                [
                    mask.sum(),
                    good[mask].sum(),
                    loss[mask].sum(),
                    b[mask].sum(),
                    conf[mask].sum(),
                    *[
                        (good[mask & (bins == i)] - conf[mask & (bins == i)]).sum()
                        for i in range(15)
                    ],
                ]
            )
        stats.append(np.asarray(rows, np.float64))
    draws = np.random.default_rng(17).integers(0, len(labels), (10000, len(labels)))
    counts = np.stack([(draws == j).sum(1) for j in range(len(labels))], 1)
    values = []
    for s in stats:
        sample = counts @ s
        values.append(
            np.column_stack(
                [sample[:, 1:5] / sample[:, :1], abs(sample[:, 5:]).sum(1) / sample[:, 0]]
            )
        )
    difference = values[1] - values[0]
    bounds = np.quantile(difference, [0.025, 0.975], axis=0, method="linear")
    return dict(
        groups=len(labels),
        group_ids=labels,
        replicates=10000,
        seed=17,
        method="whole-speaker paired percentile95%, linear quantiles",
        differences={
            k: dict(
                point=metrics[1][k] - metrics[0][k],
                lower=float(bounds[0, j]),
                upper=float(bounds[1, j]),
            )
            for j, k in enumerate(["accuracy", "nll", "brier", "mean_confidence", "ece"])
        },
        scope="observed development, correlated quartet roles; not blind",
    )
