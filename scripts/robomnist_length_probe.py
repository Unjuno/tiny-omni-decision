"""Offline source-length shortcut probe; never native modality quality."""

import hashlib
import math
import re

import numpy as np

from tiny_omni_decision.decision_math import brier_score, expected_calibration_error


def split_blocks(blocks):
    blocks = set(blocks)
    if 1 not in blocks or len(blocks) < 4 or any(type(b) is not int or b < 1 for b in blocks):
        raise ValueError("positive complete blocks including observed Rp01 required")
    ordered = sorted(
        blocks - {1},
        key=lambda b: hashlib.sha256(f"robomnist-rp-block-v1:17:{b}".encode()).hexdigest(),
    )
    held = set(ordered[: math.ceil(len(blocks) / 4)])
    return {b: "validation" if b in held else "train" for b in sorted(blocks)}


def length_features(group, sizes):
    actual = {}
    names = group["input_members"]
    if len(names) != 7 or len(set(names)) != 7:
        raise ValueError("seven distinct permitted input streams required")
    for p in names:
        stream = p.rsplit("_", 2)
        key = "_".join(stream[-2:])
        if key in actual or p not in sizes or type(sizes[p]) is not int or sizes[p] <= 0:
            raise ValueError("distinct valid positive stream size required")
        actual[key] = sizes[p]
    keys = (
        [f"Rx{i}_cam.mp4" for i in [1, 2, 3]]
        + [f"Rx{i}_mic.wav" for i in [1, 2, 3]]
        + ["Rxall_csi.json"]
    )
    if set(actual) != set(keys):
        raise ValueError("only camera/mic/CSI lengths; never positionGT or names as predictors")
    return np.log1p(np.array([actual[k] for k in keys], dtype=np.float64))


def parse_group(group_id):
    m = re.fullmatch(r"Ar([12])_V(high|med|low)_Ac([0-9])_Rp([0-9]{2})_UNC([0-9]+)", group_id)
    if not m:
        raise ValueError("full admitted physical group identity required")
    robot, velocity, action, repetition = (
        int(m[1]) - 1,
        ["high", "med", "low"].index(m[2]),
        int(m[3]),
        int(m[4]),
    )
    if repetition < 1:
        raise ValueError("positive repetition identity required")
    return velocity * 20 + robot * 10 + action, repetition


def logsumexp(a, axis=1):
    maximum = a.max(axis=axis, keepdims=True)
    return (maximum + np.log(np.exp(a - maximum).sum(axis=axis, keepdims=True))).squeeze(axis)


def centroid_logp(train_x, train_y, test_x, classes):
    if (
        train_x.ndim != 2
        or test_x.ndim != 2
        or not len(train_x)
        or not len(test_x)
        or train_x.shape[1] != test_x.shape[1]
        or not train_x.shape[1]
        or train_y.shape != (len(train_x),)
        or not np.issubdtype(train_y.dtype, np.integer)
        or set(train_y) != set(range(classes))
        or not np.isfinite(train_x).all()
        or not np.isfinite(test_x).all()
    ):
        raise ValueError("finite aligned train/validation inputs and every training class required")
    mean, scale = train_x.mean(0), train_x.std(0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    standardized = (train_x - mean) / scale
    centroids = np.stack([standardized[train_y == c].mean(0) for c in range(classes)])
    logits = -0.5 * np.square(((test_x - mean) / scale)[:, None, :] - centroids[None]).sum(2)
    if not np.isfinite(logits).all():
        raise ValueError("nonfinite numeric shortcut logits")
    return logits - logsumexp(logits)[:, None]


def task_mapping(task):
    c = np.arange(60)
    return {"activity60": c, "robot": (c % 20) // 10, "velocity": c // 20, "trajectory": c % 10}[
        task
    ]


def task_logp(joint_logp, task):
    if joint_logp.ndim != 2 or joint_logp.shape[1] != 60 or not np.isfinite(joint_logp).all():
        raise ValueError("complete finite60-class posterior required")
    mapping = task_mapping(task)
    return np.stack(
        [logsumexp(joint_logp[:, mapping == c]) for c in range(int(mapping.max()) + 1)], 1
    )


def metrics(logp, targets):
    if (
        logp.ndim != 2
        or not len(logp)
        or logp.shape[1] < 2
        or targets.shape != (len(logp),)
        or not np.issubdtype(targets.dtype, np.integer)
        or (targets < 0).any()
        or (targets >= logp.shape[1]).any()
        or not np.isfinite(logp).all()
        or not np.allclose(logsumexp(logp), 0, atol=1e-12, rtol=0)
    ):
        raise ValueError("complete normalized finite stable log posterior and targets required")
    p = np.exp(logp)
    return dict(
        count=len(targets),
        accuracy=float((p.argmax(1) == targets).mean()),
        nll=float(-logp[np.arange(len(targets)), targets].mean()),
        brier=brier_score(p.tolist(), targets.tolist()),
        ece=expected_calibration_error(p.tolist(), targets.tolist(), n_bins=15),
        mean_confidence=float(p.max(1).mean()),
    )


def paired_block_interval(correct, control, repetitions, *, draws=2000, seed=17):
    blocks = sorted(set(repetitions.tolist()))
    diffs = [
        float((correct[repetitions == b].astype(float) - control[repetitions == b]).sum())
        for b in blocks
    ]
    counts = [int((repetitions == b).sum()) for b in blocks]
    rng = np.random.default_rng(seed)
    selected = rng.integers(0, len(blocks), (draws, len(blocks)))
    delta = np.array(diffs)[selected].sum(1) / np.array(counts)[selected].sum(1)
    return dict(
        delta_accuracy=float((correct.astype(float) - control).mean()),
        interval_95_percentile=np.quantile(delta, [0.025, 0.975]).tolist(),
        draws=draws,
        seed=seed,
        repetition_blocks=len(blocks),
        method="paired complete-Rp cluster percentile bootstrap; development only",
    )
