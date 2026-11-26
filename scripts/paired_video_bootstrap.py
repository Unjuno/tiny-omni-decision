"""Fixed development diagnostic; rows must be ordered initial/final video pairs."""

import numpy as np


def paired_video_bootstrap(normal, mismatch, targets, *, probability_floor=1e-15):
    """Normal minus mismatch, resampling whole videos (not independent questions)."""
    if not np.isfinite(probability_floor) or not 0 < probability_floor < 1:
        raise ValueError("probability floor must be finite and between zero and one")
    normal = np.asarray(normal, dtype=np.float64)
    mismatch = np.asarray(mismatch, dtype=np.float64)
    targets = np.asarray(targets)
    if (
        normal.ndim != 2
        or normal.shape != mismatch.shape
        or normal.shape[0] == 0
        or normal.shape[0] % 2
        or normal.shape[1] < 2
        or targets.shape != (normal.shape[0],)
        or targets.dtype.kind not in "iu"
        or np.any(targets < 0)
        or np.any(targets >= normal.shape[1])
    ):
        raise ValueError("complete ordered video pairs and valid targets required")
    for value in (normal, mismatch):
        if (
            not np.isfinite(value).all()
            or np.any(value < 0)
            or np.any(value > 1)
            or not np.allclose(value.sum(-1), 1, atol=1e-6, rtol=0)
        ):
            raise ValueError("invalid probabilities")
    rows = np.arange(len(targets))
    differences = {
        "accuracy": (normal.argmax(-1) == targets).astype(float)
        - (mismatch.argmax(-1) == targets).astype(float),
        "nll": -np.log(np.clip(normal[rows, targets], probability_floor, 1))
        + np.log(np.clip(mismatch[rows, targets], probability_floor, 1)),
    }
    rng = np.random.default_rng(17)
    indices = rng.integers(len(targets) // 2, size=(10000, len(targets) // 2))
    result = dict(
        seed=17, replicates=10000, unit="video_pair", videos=len(targets) // 2,
        method="95% percentile; numpy linear quantile", direction="normal-minus-mismatch",
        probability_floor=probability_floor,
    )
    for name, difference in differences.items():
        per_video = difference.reshape(-1, 2).mean(-1)
        samples = per_video[indices].mean(-1)
        result[name] = dict(
            difference=float(per_video.mean()),
            interval=np.quantile(samples, [0.025, 0.975], method="linear").tolist(),
        )
    return result
