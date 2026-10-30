"""Paired video-cluster bootstrap; questions within each video stay together."""

import numpy as np


def _statistics(probabilities, targets):
    # [draw, video, condition, option], with two conditions per video.
    correct = probabilities.argmax(-1) == targets
    confidence = probabilities.max(-1)
    selected = np.take_along_axis(probabilities, targets[..., None], -1)[..., 0]
    onehot = np.eye(probabilities.shape[-1])[targets]
    result = dict(
        accuracy=correct.mean((1, 2)),
        nll=-np.log(np.maximum(selected, 1e-12)).mean((1, 2)),
        brier=((probabilities - onehot) ** 2).sum(-1).mean((1, 2)),
        both_correct=correct.all(-1).mean(1),
        initial_accuracy=correct[:, :, 0].mean(1),
        final_accuracy=correct[:, :, 1].mean(1),
    )
    bins = np.minimum((confidence * 15).astype(int), 14)
    result["ece"] = sum(
        np.abs(((correct - confidence) * (bins == index)).sum((1, 2)))
        / (targets.shape[1] * targets.shape[2])
        for index in range(15)
    )
    return result


def video_cluster_intervals(probabilities, targets, *, reference=None, draws=10000, seed=17):
    """Percentile95 intervals; optional paired differences use identical draws."""
    probabilities = np.asarray(probabilities, dtype=float)
    targets = np.asarray(targets)
    if probabilities.ndim != 3 or probabilities.shape[1] != 2:
        raise ValueError("expected [video,2,options]")
    if targets.shape != probabilities.shape[:2] or not len(targets):
        raise ValueError("aligned video/condition targets required")
    if not np.issubdtype(targets.dtype, np.integer):
        raise ValueError("integer targets required")
    if np.any(targets < 0) or np.any(targets >= probabilities.shape[-1]):
        raise ValueError("invalid targets")
    if not isinstance(draws, int) or draws < 1:
        raise ValueError("positive draw count required")
    arrays = [probabilities]
    if reference is not None:
        reference = np.asarray(reference, dtype=float)
        if reference.shape != probabilities.shape:
            raise ValueError("aligned reference required")
        arrays.append(reference)
    for value in arrays:
        if not np.isfinite(value).all() or np.any(value < 0):
            raise ValueError("invalid probabilities")
        if not np.allclose(value.sum(-1), 1, atol=1e-6):
            raise ValueError("probabilities must sum to one")
    generator = np.random.default_rng(seed)
    samples, differences = {}, {}
    for offset in range(0, draws, 100):
        indices = generator.integers(len(targets), size=(min(100, draws - offset), len(targets)))
        score = _statistics(probabilities[indices], targets[indices])
        baseline = (
            _statistics(reference[indices], targets[indices]) if reference is not None else None
        )
        for name, value in score.items():
            samples.setdefault(name, []).extend(value.tolist())
            if baseline is not None:
                differences.setdefault(name, []).extend((value - baseline[name]).tolist())
    result = dict(
        draws=draws,
        seed=seed,
        clusters=len(targets),
        method="video-cluster resampling with replacement; percentile95 [2.5,97.5]",
        intervals={
            name: np.percentile(value, [2.5, 97.5]).tolist() for name, value in samples.items()
        },
    )
    if reference is not None:
        result["paired_difference_intervals"] = {
            name: np.percentile(value, [2.5, 97.5]).tolist() for name, value in differences.items()
        }
    return result
