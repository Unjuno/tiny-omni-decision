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


def fit_l2_softmax(features: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    """Fit a deterministic L2 multinomial logistic head; standardization uses train only."""
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.int64)
    if x.ndim != 2 or y.shape != (len(x),) or len(x) < 2:
        raise ValueError("expected a feature matrix and one target per row")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("features and targets must be finite")
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
        value = float(loss + 0.5 * np.square(weights).sum() / count)
        residual = probabilities
        residual[np.arange(count), y] -= 1.0
        residual /= count
        gradient_weights = residual.T @ z + weights / count
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
        "converged": True,
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
