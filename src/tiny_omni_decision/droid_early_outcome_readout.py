"""Small fixed binary readouts for the exploratory DROID outcome task."""

from __future__ import annotations

from typing import Any

import numpy as np


def fit_l2_logistic(features: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    """Fit one deterministic L2 logistic readout, standardizing on train rows only."""
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if x.ndim != 2 or y.shape != (len(x),) or len(x) < 2:
        raise ValueError("expected a feature matrix and one target per row")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or set(np.unique(y)) != {0.0, 1.0}:
        raise ValueError("features must be finite and targets must contain both binary classes")

    from scipy.optimize import minimize
    from scipy.special import expit

    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-6] = 1.0
    z = np.clip((x - mean) / scale, -20, 20)
    count, width = z.shape

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        weights = parameters[:width]
        bias = parameters[width]
        logits = z @ weights + bias
        losses = np.logaddexp(0.0, logits) - y * logits
        value = float(losses.mean() + 0.5 * np.dot(weights, weights) / count)
        residual = (expit(logits) - y) / count
        gradient = np.concatenate((z.T @ residual + weights / count, [residual.sum()]))
        return value, gradient

    result = minimize(
        objective,
        np.zeros(width + 1),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8, "maxls": 40},
    )
    if not result.success:
        raise RuntimeError(f"L2 logistic optimization did not converge: {result.message}")
    return {
        "mean": mean,
        "scale": scale,
        "weight": result.x[:width],
        "bias": float(result.x[width]),
        "objective": float(result.fun),
        "iterations": int(result.nit),
        "converged": True,
    }


def predict_success_probability(model: dict[str, Any], features: np.ndarray) -> np.ndarray:
    from scipy.special import expit

    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != len(model["mean"]):
        raise ValueError("feature width differs from the fitted readout")
    if not np.isfinite(x).all():
        raise ValueError("prediction features must be finite")
    z = np.clip((x - model["mean"]) / model["scale"], -20, 20)
    return expit(z @ model["weight"] + model["bias"])


def binary_probability_metrics(targets: np.ndarray, p_success: np.ndarray) -> dict[str, Any]:
    """Return binary accuracy and calibration metrics; ECE uses ten fixed bins."""
    y = np.asarray(targets, dtype=np.int64)
    p = np.asarray(p_success, dtype=np.float64)
    if y.ndim != 1 or p.shape != y.shape or len(y) == 0:
        raise ValueError("targets and probabilities must be non-empty aligned vectors")
    if not set(np.unique(y)).issubset({0, 1}) or not np.isfinite(p).all():
        raise ValueError("targets must be binary and probabilities finite")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("probabilities must lie in [0, 1]")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("both target classes are required for balanced metrics")

    clipped = np.clip(p, 1e-12, 1 - 1e-12)
    prediction = (clipped >= 0.5).astype(np.int64)
    correct = prediction == y
    confidence = np.maximum(clipped, 1 - clipped)
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
        "balanced_accuracy": float((correct[y == 0].mean() + correct[y == 1].mean()) / 2),
        "nll": float(np.mean(-y * np.log(clipped) - (1 - y) * np.log(1 - clipped))),
        "brier": float(np.mean((clipped - y) ** 2)),
        "ece": ece,
        "mean_confidence": float(confidence.mean()),
        "failure_recall": float((prediction[y == 0] == 0).mean()),
        "success_recall": float((prediction[y == 1] == 1).mean()),
    }
