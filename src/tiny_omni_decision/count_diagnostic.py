"""Training-side count errors and fixed-candidate readout algebra."""

import numpy as np


def count_errors(targets, predictions, *, classes):
    a, b = np.asarray(targets), np.asarray(predictions)
    if (
        a.ndim != 1
        or not len(a)
        or a.shape != b.shape
        or not np.isin(a, classes).all()
        or not np.isin(b, classes).all()
        or len(set(classes)) != len(classes)
    ):
        raise ValueError("nonempty paired integer class labels required")
    if not np.equal(a, a.astype(int)).all() or not np.equal(b, b.astype(int)).all():
        raise ValueError("integer counts required")
    indices = {v: i for i, v in enumerate(classes)}
    confusion = np.zeros((len(classes), len(classes)), dtype=int)
    for x, y in zip(a, b, strict=True):
        confusion[indices[x], indices[y]] += 1
    error = b - a
    return dict(
        count=len(a),
        classes=list(classes),
        exact_accuracy=float((error == 0).mean()),
        underestimated=int((error < 0).sum()),
        overestimated=int((error > 0).sum()),
        mean_absolute_error=float(np.abs(error).mean()),
        signed_error=float(error.mean()),
        absolute_error_histogram={
            str(int(v)): int((np.abs(error) == v).sum()) for v in np.unique(np.abs(error))
        },
        confusion_true_by_predicted=confusion.tolist(),
    )


def fixed_candidate_affine(model, candidates):
    import torch

    if (
        candidates.ndim != 2
        or len(candidates) < 2
        or candidates.shape[1] != model.candidate.in_features
        or not torch.isfinite(candidates).all()
    ):
        raise ValueError("finite fixed candidate matrix required")
    with torch.no_grad():
        projected = model.candidate(candidates)
        weight = projected @ model.observation.weight / model.scale
        bias = projected @ model.observation.bias / model.scale
    return weight, bias


def fixed_taxonomy_loss(logits, targets):
    from torch.nn import functional as functional

    if logits.ndim != 3 or targets.shape != logits.shape[:2]:
        raise ValueError("[image,task,class] logits and aligned [image,task] targets required")
    return functional.cross_entropy(logits.flatten(0, 1), targets.flatten())
