"""Audit completed fit timelines and the existing selector; no file/model I/O."""

import math
from collections import Counter

from scripts.decision_semantic_training import learning_rate
from scripts.decision_supervision import digest, select_validation


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, expected):
    return type(value) is int and value == expected


def audit_fit_evidence(run, history, updates, order, schema):
    """Recompute execution/selection evidence after the caller has bound all bytes."""
    _require(run.get("status") == "PASS_MATCHED_RELEVANCE_FIT", "completed fit required")
    _require(len(updates) == len(order) == 1024, "complete1024 updates and sampler required")
    config = run["config"]
    positive_lr, total_seconds = 0, 0.0
    auxiliary_gradients = []
    for step, row in enumerate(updates, 1):
        _require(_integer(row.get("step"), step), "update step sequence changed")
        _require(_integer(row.get("consumed"), 18 * step), "update consumption changed")
        expected = learning_rate(step, 1024, config["warmup_updates"], config["learning_rate"])
        _require(row.get("learning_rate") == expected, "learning-rate timeline changed")
        for key in (
            "ce",
            "kl",
            "relevance_loss",
            "gradient_norm",
            "auxiliary_gradient_norm_after_clip",
            "seconds",
        ):
            value = row.get(key)
            _require(
                type(value) in (int, float) and math.isfinite(value),
                "nonfinite/missing update " + key,
            )
        _require(row["seconds"] >= 0 and row["gradient_norm"] >= 0, "invalid timing/gradient")
        _require(row["auxiliary_gradient_norm_after_clip"] >= 0, "invalid auxiliary gradient")
        auxiliary_gradients.append(row["auxiliary_gradient_norm_after_clip"])
        positive_lr += expected > 0
        total_seconds += row["seconds"]
    _require(run["arm"] in ("ce", "relevance"), "registered arm required")
    _require(
        all(g == 0 for g in auxiliary_gradients)
        if run["arm"] == "ce"
        else any(g > 0 for g in auxiliary_gradients),
        "auxiliary gradient does not match intervention",
    )
    seen = [set(), set(), set()]
    for batch in order:
        _require(
            isinstance(batch, list) and len(batch) == 18, "complete18-question update required"
        )
        counts = Counter()
        for item in batch:
            _require(
                isinstance(item, list) and len(item) == 2 and all(type(v) is int for v in item),
                "integer sampler pair required",
            )
            bucket, index = item
            _require(bucket in (0, 1, 2), "unregistered sampler bucket")
            _require(
                0 <= index < (8192, 8192, 2048)[bucket] and index not in seen[bucket],
                "sampler duplicate/out-of-range index",
            )
            seen[bucket].add(index)
            counts[bucket] += 1
        _require(counts == {0: 8, 1: 8, 2: 2}, "realized sampling ratio changed")
    _require([len(v) for v in seen] == [8192, 8192, 2048], "incomplete sampler coverage")
    _require(len(schema) == 47, "all47 metric slices required")
    _require(
        [r.get("step") for r in history] == [0, 256, 512, 768, 1024],
        "scheduled validation history required",
    )
    baseline = history[0]["metrics"]
    curves = []
    for index, row in enumerate(history):
        best = select_validation(history[: index + 1], baseline, schema)
        _require(_integer(row.get("best_step"), best), "stored selector state changed")
        consumed = 18 * row["step"]
        _require(
            all(
                _integer(row["consumed"].get(k), expected)
                for k, expected in (
                    ("consumed", consumed),
                    ("unique_questions", consumed),
                    ("repeated_questions", 0),
                )
            ),
            "evaluation accounting changed",
        )
        recomputed = {
            key: sum(m[key] for m in row["metrics"].values()) / 47
            for key in ("accuracy", "nll", "brier", "ece")
        }
        _require(
            all(
                math.isclose(row["macro"][k], v, rel_tol=0, abs_tol=1e-12)
                for k, v in recomputed.items()
            ),
            "macro metric mismatch",
        )
        _require(
            row["minimum_accuracy"] == min(m["accuracy"] for m in row["metrics"].values()),
            "minimum metric mismatch",
        )
        curves.append(
            dict(
                step=row["step"],
                best_step=best,
                consumed_questions=consumed,
                macro=recomputed,
                minimum_accuracy=row["minimum_accuracy"],
            )
        )
    best = curves[-1]["best_step"]
    _require(_integer(run.get("best_step"), best), "terminal selector differs from recomputation")
    _require(_integer(run["consumed"]["consumed"], 18432), "terminal full-run count changed")
    _require(
        _integer(run["selected_consumed"]["consumed"], best * 18), "terminal selected count changed"
    )
    for part in ("validation", "fixed-train"):
        for suffix in ("scores.npy", "metrics.json", "predictions.jsonl"):
            original = run["artifacts"].get(f"{part}-{best}-{suffix}")
            replay = run["artifacts"].get(f"reload-{part}-{suffix}")
            _require(
                isinstance(original, str) and len(original) == 64 and original == replay,
                "selected reload artifact mismatch",
            )
    return dict(
        updates=1024,
        positive_lr_updates=positive_lr,
        consumed_questions=18432,
        best_step=best,
        selected_consumed_questions=best * 18,
        bucket_counts=[len(v) for v in seen],
        repeated_question_indices=0,
        update_seconds=total_seconds,
        lr_consumption_sha256=digest(
            [[u["step"], u["learning_rate"], u["consumed"]] for u in updates]
        ),
        sampler_sha256=digest(order),
        curves=curves,
        selected_reload_artifacts_exact=True,
    )
