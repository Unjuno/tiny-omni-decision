"""Fixed multiclass frozen-feature readout for DROID next-action direction."""

from __future__ import annotations

from typing import Any

import numpy as np

ACTION_OPTIONS = ("stop", "x-", "x+", "y-", "y+", "z-", "z+")


def next_cartesian_direction_label(
    velocity: np.ndarray, stop_threshold: float = 1e-4
) -> int:
    """Map a 6D commanded Cartesian velocity to stop or dominant signed translation axis."""
    values = np.asarray(velocity, dtype=np.float64)
    if values.shape != (6,) or not np.isfinite(values).all():
        raise ValueError("velocity must contain six finite Cartesian components")
    if stop_threshold < 0:
        raise ValueError("stop threshold must be non-negative")
    translation = values[:3]
    if float(np.max(np.abs(translation))) <= stop_threshold:
        return 0
    axis = int(np.argmax(np.abs(translation)))
    return 1 + axis * 2 + int(translation[axis] > 0)


def make_readout_features(video: np.ndarray, sensor: np.ndarray) -> dict[str, np.ndarray]:
    """Build fixed-width temporal summaries without mixing modality branches."""
    video_array = np.asarray(video, dtype=np.float64)
    sensor_array = np.asarray(sensor, dtype=np.float64)
    if video_array.ndim != 4 or video_array.shape[1:3] != (3, 8):
        raise ValueError("video features must have shape [N, 3 views, 8 times, D]")
    if sensor_array.ndim != 3 or sensor_array.shape[1:] != (8, 14):
        raise ValueError("sensor features must have shape [N, 8 times, 14]")
    if len(video_array) != len(sensor_array):
        raise ValueError("video and sensor batches must align")
    if not np.isfinite(video_array).all() or not np.isfinite(sensor_array).all():
        raise ValueError("features must be finite")
    temporal_video = video_array.mean(axis=1)
    video_summary = np.concatenate(
        (temporal_video[:, :4].mean(axis=1), temporal_video[:, 4:].mean(axis=1)), axis=1
    )
    sensor_summary = sensor_array.reshape(len(sensor_array), -1)
    return {
        "video": video_summary,
        "sensor": sensor_summary,
        "video_sensor": np.concatenate((video_summary, sensor_summary), axis=1),
    }


def fit_l2_softmax(
    features: np.ndarray, targets: np.ndarray, l2_strength: float = 1.0
) -> dict[str, Any]:
    """Fit a deterministic L2 multinomial logistic head; standardization uses train only."""
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.int64)
    if x.ndim != 2 or y.shape != (len(x),) or len(x) < 2:
        raise ValueError("expected a feature matrix and one target per row")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("features and targets must be finite")
    if not np.isfinite(l2_strength) or l2_strength <= 0:
        raise ValueError("L2 strength must be a positive finite value")
    if np.any((y < 0) | (y >= len(ACTION_OPTIONS))) or len(np.unique(y)) < 2:
        raise ValueError("targets must include at least two valid action classes")

    from scipy.optimize import minimize
    from scipy.special import softmax

    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-6] = 1.0
    z = np.clip((x - mean) / scale, -20, 20)
    count, width = z.shape
    classes = len(ACTION_OPTIONS)

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        weights = parameters[: classes * width].reshape(classes, width)
        bias = parameters[classes * width :]
        logits = z @ weights.T + bias
        probabilities = softmax(logits, axis=1)
        loss = -np.log(np.maximum(probabilities[np.arange(count), y], 1e-15)).mean()
        value = float(loss + 0.5 * l2_strength * np.square(weights).sum() / count)
        residual = probabilities
        residual[np.arange(count), y] -= 1.0
        residual /= count
        gradient_weights = residual.T @ z + l2_strength * weights / count
        gradient_bias = residual.sum(axis=0)
        return value, np.concatenate((gradient_weights.ravel(), gradient_bias))

    initial = np.zeros(classes * width + classes, dtype=np.float64)
    result = minimize(
        objective,
        initial,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8, "maxls": 40},
    )
    if not result.success:
        raise RuntimeError(f"multiclass L2 logistic optimization failed: {result.message}")
    return {
        "mean": mean,
        "scale": scale,
        "weight": result.x[: classes * width].reshape(classes, width),
        "bias": result.x[classes * width :],
        "objective": float(result.fun),
        "iterations": int(result.nit),
        "l2_strength": float(l2_strength),
        "converged": True,
    }


def stratified_group_folds(
    targets: np.ndarray,
    group_ids: list[str] | tuple[str, ...],
    folds: int = 5,
    seed: int = 17,
) -> list[np.ndarray]:
    """Create deterministic class-stratified folds without splitting source groups."""
    y = np.asarray(targets, dtype=np.int64)
    groups = np.asarray(group_ids, dtype=str)
    if y.ndim != 1 or groups.shape != y.shape or folds < 2:
        raise ValueError("targets, group IDs, and fold count are invalid")
    labels_by_group: dict[str, set[int]] = {}
    rows_by_group: dict[str, list[int]] = {}
    for index, (label, group) in enumerate(zip(y, groups, strict=True)):
        labels_by_group.setdefault(str(group), set()).add(int(label))
        rows_by_group.setdefault(str(group), []).append(index)
    if any(len(labels) != 1 for labels in labels_by_group.values()):
        raise ValueError("each group must have one target class for stratified allocation")
    rows_by_class: dict[int, list[str]] = {}
    for group, labels in labels_by_group.items():
        rows_by_class.setdefault(next(iter(labels)), []).append(group)
    rng = np.random.default_rng(seed)
    fold_groups: list[list[str]] = [[] for _ in range(folds)]
    for label in sorted(rows_by_class):
        class_groups = sorted(rows_by_class[label])
        if len(class_groups) < folds:
            raise ValueError("each class must contain at least one group per fold")
        order = rng.permutation(len(class_groups))
        for offset, group_index in enumerate(order):
            fold_groups[offset % folds].append(class_groups[int(group_index)])
    result = [
        np.asarray(
            sorted(index for group in groups_for_fold for index in rows_by_group[group]),
            dtype=np.int64,
        )
        for groups_for_fold in fold_groups
    ]
    all_indices = np.concatenate(result)
    if len(all_indices) != len(y) or not np.array_equal(np.sort(all_indices), np.arange(len(y))):
        raise RuntimeError("stratified folds do not partition every row exactly once")
    return result


def select_l2_strength_group_cv(
    features: np.ndarray,
    targets: np.ndarray,
    group_ids: list[str] | tuple[str, ...],
    candidates: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0),
    folds: int = 5,
    seed: int = 17,
) -> dict[str, Any]:
    """Choose L2 strength by out-of-fold macro NLL on training groups only."""
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.int64)
    if x.ndim != 2 or y.shape != (len(x),) or len(group_ids) != len(y):
        raise ValueError("training features, targets, and groups must align")
    if not candidates or any(not np.isfinite(value) or value <= 0 for value in candidates):
        raise ValueError("L2 candidates must be positive finite values")
    validation_folds = stratified_group_folds(y, group_ids, folds=folds, seed=seed)
    candidate_metrics = {}
    for strength in candidates:
        out_of_fold = np.zeros((len(y), len(ACTION_OPTIONS)), dtype=np.float64)
        covered = np.zeros(len(y), dtype=bool)
        for validation_index in validation_folds:
            train_mask = np.ones(len(y), dtype=bool)
            train_mask[validation_index] = False
            model = fit_l2_softmax(x[train_mask], y[train_mask], l2_strength=strength)
            out_of_fold[validation_index] = predict_softmax(model, x[validation_index])
            covered[validation_index] = True
        if not covered.all():
            raise RuntimeError("cross-validation did not predict every training row")
        candidate_metrics[str(strength)] = multiclass_probability_metrics(y, out_of_fold)
    selected = min(
        candidates,
        key=lambda strength: (candidate_metrics[str(strength)]["macro_nll"], -strength),
    )
    return {
        "selected_l2_strength": float(selected),
        "folds": int(folds),
        "seed": int(seed),
        "selection_metric": "out-of-fold macro NLL",
        "candidate_metrics": candidate_metrics,
    }


def predict_softmax(model: dict[str, Any], features: np.ndarray) -> np.ndarray:
    from scipy.special import softmax

    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != len(model["mean"]):
        raise ValueError("feature width differs from the fitted readout")
    if not np.isfinite(x).all():
        raise ValueError("prediction features must be finite")
    z = np.clip((x - model["mean"]) / model["scale"], -20, 20)
    return softmax(z @ model["weight"].T + model["bias"], axis=1)


def multiclass_probability_metrics(
    targets: np.ndarray, probabilities: np.ndarray
) -> dict[str, Any]:
    """Compute accuracy, macro metrics, Brier, and confidence ECE for fixed options."""
    y = np.asarray(targets, dtype=np.int64)
    p = np.asarray(probabilities, dtype=np.float64)
    if y.ndim != 1 or p.shape != (len(y), len(ACTION_OPTIONS)) or len(y) == 0:
        raise ValueError("targets and probabilities have incompatible shapes")
    if np.any((y < 0) | (y >= len(ACTION_OPTIONS))) or not np.isfinite(p).all():
        raise ValueError("targets or probabilities are invalid")
    if np.any(p < 0) or not np.allclose(p.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("probabilities must be non-negative and sum to one")
    predictions = p.argmax(axis=1)
    confidence = p.max(axis=1)
    correct = predictions == y
    one_hot = np.eye(len(ACTION_OPTIONS), dtype=np.float64)[y]
    per_class = {}
    for class_id, name in enumerate(ACTION_OPTIONS):
        mask = y == class_id
        if mask.any():
            per_class[name] = {
                "count": int(mask.sum()),
                "accuracy": float((predictions[mask] == class_id).mean()),
                "nll": float(-np.log(np.maximum(p[mask, class_id], 1e-15)).mean()),
            }
    ece = 0.0
    for lower in np.linspace(0.0, 1.0, 11)[:-1]:
        upper = lower + 0.1
        mask = (confidence >= lower) & (
            (confidence < upper) if upper < 1.0 else (confidence <= upper)
        )
        if mask.any():
            ece += float(mask.mean()) * abs(float(correct[mask].mean() - confidence[mask].mean()))
    return {
        "count": len(y),
        "accuracy": float(correct.mean()),
        "macro_accuracy": float(np.mean([m["accuracy"] for m in per_class.values()])),
        "nll": float(-np.log(np.maximum(p[np.arange(len(y)), y], 1e-15)).mean()),
        "macro_nll": float(np.mean([m["nll"] for m in per_class.values()])),
        "brier": float(np.square(p - one_hot).sum(axis=1).mean()),
        "ece_10": ece,
        "mean_confidence": float(confidence.mean()),
        "per_class": per_class,
    }
