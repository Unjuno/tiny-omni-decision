"""Fixed role-membership metrics and controls for the QA experiment."""

import hashlib

import numpy as np


def arm_input(points, arm):
    if arm == "trajectory":
        return points
    if arm != "initial_only":
        raise ValueError("unknown arm")
    result = points.clone()
    result[..., 4:128] = 0
    return result


def membership_metrics(probabilities, targets):
    p, y = np.asarray(probabilities, dtype=np.float64), np.asarray(targets)
    if p.ndim != 2 or min(p.shape) == 0 or y.shape != p.shape or y.dtype != bool:
        raise ValueError("nonempty aligned Boolean membership required")
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("finite probabilities required")
    clipped = np.clip(p, 1e-12, 1 - 1e-12)
    loss = -np.where(y, np.log(clipped), np.log1p(-clipped))
    positive, negative = y.sum(1), (~y).sum(1)
    pos = (loss * y).sum(1) / np.maximum(positive, 1)
    neg = (loss * ~y).sum(1) / np.maximum(negative, 1)
    classes = (positive > 0).astype(int) + (negative > 0).astype(int)
    selected = np.argmax(p, axis=1)
    hits = y[np.arange(len(y)), selected]
    prediction = p >= 0.5
    confidence = np.maximum(p, 1 - p)
    correct = prediction == y
    ece = 0.0
    for i in range(15):
        mask = (confidence >= i / 15) & (confidence <= 1 if i == 14 else confidence < (i + 1) / 15)
        if mask.any():
            ece += mask.mean() * abs(confidence[mask].mean() - correct[mask].mean())
    return dict(
        clips=len(p),
        points=int(p.size),
        grid_supported_clips=int((positive > 0).sum()),
        top1_point_in_cover_bbox=float(hits.mean()),
        top1_hits=hits.tolist(),
        balanced_binary_nll=float(((pos + neg) / classes).mean()),
        natural_grid_accuracy=float(correct.mean()),
        nll=float(loss.mean()),
        brier=float(2 * np.square(p - y).mean()),
        ece15=float(ece),
        positive_fraction=float(y.mean()),
        positive_recall=float(prediction[y].mean()) if y.any() else None,
        negative_recall=float((~prediction[~y]).mean()) if (~y).any() else None,
    )


def state_hash(model):
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
