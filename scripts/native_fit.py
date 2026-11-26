"""Fixed sampler, metrics and research selector for common native supervision."""

import hashlib
import math
import random
from collections import Counter, defaultdict

import numpy as np

from scripts.semantic_state_diagnostic import metrics


def sample_order(rows, steps, seed=17):
    if (
        type(steps) is not int
        or steps < 1
        or not rows
        or any(r["split"] != "train" for r in rows)
        or len({r["id"] for r in rows}) != len(rows)
    ):
        raise ValueError("unique TRAIN questions and positive steps required")
    pools = defaultdict(list)
    for i, row in enumerate(rows):
        pools[row["modality"], row["cohort"], row["task"]].append(i)
    modalities = ("text", "image", "audio", "video")
    if {k[0] for k in pools} != set(modalities):
        raise ValueError("all four modalities required")
    cohorts = {m: sorted({k[1] for k in pools if k[0] == m}) for m in modalities}
    tasks = {
        (m, c): sorted({k[2] for k in pools if k[:2] == (m, c)})
        for m in modalities
        for c in cohorts[m]
    }
    rngs = {
        k: random.Random(int.from_bytes(hashlib.sha256(f"{seed}:{k}".encode()).digest()[:8], "big"))
        for k in pools
    }
    for key, pool in pools.items():
        rngs[key].shuffle(pool)
    offsets, visits = Counter(), Counter()
    result = []
    for step in range(steps):
        batch = []
        for mod in modalities:
            cohort = cohorts[mod][step % len(cohorts[mod])]
            task = tasks[mod, cohort][visits[mod, cohort] % len(tasks[mod, cohort])]
            visits[mod, cohort] += 1
            key = mod, cohort, task
            if offsets[key] == len(pools[key]):
                rngs[key].shuffle(pools[key])
                offsets[key] = 0
            batch.append(pools[key][offsets[key]])
            offsets[key] += 1
        result.append(batch)
    return result


def option_order(row, step, seed=17):
    digest = hashlib.sha256(f"{seed}:{step}:{row['id']}".encode()).digest()
    order = list(range(len(row["options"])))
    random.Random(int.from_bytes(digest[:8], "big")).shuffle(order)
    return order


def accounting(rows):
    def asset(row):
        ident = row["observation"]
        return ident.rsplit(":", 1)[0] if row["source"] == "starss" else ident

    return dict(
        consumed=len(rows),
        unique_questions=len({r["id"] for r in rows}),
        repeated_questions=len(rows) - len({r["id"] for r in rows}),
        unique_observations=len({r["observation"] for r in rows}),
        unique_media_assets=len({asset(r) for r in rows if r["modality"] != "text"}),
        unique_groups=len({(r["modality"], r["group"]) for r in rows}),
        modalities=dict(Counter(r["modality"] for r in rows)),
        sources=dict(Counter(r["source"] for r in rows)),
        tasks=dict(Counter(r["source"] + ":" + r["task"] for r in rows)),
    )


def learning_rate(step):
    if type(step) is not int or not 1 <= step <= 2048:
        raise ValueError("fixed2048-step schedule required")
    if step <= 62:
        return 0.001 * step / 62
    return 0.001 * 0.5 * (1 + math.cos(math.pi * (step - 62) / (2048 - 62)))


def choose_best(best, candidate):
    if not math.isfinite(candidate) or (best is not None and not math.isfinite(best)):
        raise ValueError("finite selection scores required")
    return best is None or candidate < best - 1e-5


def summarize(rows, scores):
    scores = np.asarray(scores)
    if (
        not rows
        or scores.ndim != 2
        or len(rows) != len(scores)
        or not np.isfinite(scores).all()
        or any(
            not 0 <= r["target_index"] < len(r["options"]) <= scores.shape[1]
            or np.any(scores[i, len(r["options"]) :] > -1e29)
            for i, r in enumerate(rows)
        )
    ):
        raise ValueError(
            "finite complete scores, exact supplied options and valid targets required"
        )
    target = np.array([r["target_index"] for r in rows])

    def grouped(keys):
        groups = defaultdict(list)
        for i, r in enumerate(rows):
            groups[":".join(r[k] for k in keys)].append(i)
        return {key: metrics(scores[index], target[index]) for key, index in sorted(groups.items())}

    tasks = grouped(("source", "task"))
    sources = grouped(("source",))
    modalities = grouped(("modality",))
    names = ("accuracy", "nll", "brier", "ece")
    task_keys = {
        s: sorted({r["source"] + ":" + r["task"] for r in rows if r["source"] == s})
        for s in sources
    }
    source_macro = {
        s: {k: float(np.mean([tasks[t][k] for t in ts])) for k in names}
        for s, ts in task_keys.items()
    }
    mod_sources = {m: sorted({r["source"] for r in rows if r["modality"] == m}) for m in modalities}
    mod_macro = {
        m: {k: float(np.mean([source_macro[s][k] for s in ss])) for k in names}
        for m, ss in mod_sources.items()
    }
    events = {}
    for task in sorted({r["task"] for r in rows if r["source"] == "starss"}):
        indices = [i for i, r in enumerate(rows) if r["source"] == "starss" and r["task"] == task]
        truth = target[indices]
        prediction = scores[indices].argmax(1)
        positive = truth == 1
        negative = truth == 0
        recall = float((prediction[positive] == 1).mean()) if positive.any() else None
        specificity = float((prediction[negative] == 0).mean()) if negative.any() else None
        events[task] = dict(
            positive_count=int(positive.sum()),
            negative_count=int(negative.sum()),
            recall=recall,
            specificity=specificity,
            balanced_accuracy=(
                (recall + specificity) / 2
                if recall is not None and specificity is not None
                else None
            ),
        )
    return dict(
        overall=metrics(scores, target),
        modalities=modalities,
        sources=sources,
        tasks=tasks,
        source_macro=source_macro,
        modality_macro=mod_macro,
        macro={k: float(np.mean([v[k] for v in mod_macro.values()])) for k in names},
        minimum_modality_accuracy=min(v["accuracy"] for v in mod_macro.values()),
        minimum_task_accuracy=min(v["accuracy"] for v in tasks.values()),
        event_classes=events,
    )
