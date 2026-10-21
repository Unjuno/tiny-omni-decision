"""Descriptive matched-prediction diagnostics, never a label-aware inference rule."""

import numpy as np


def complementarity(first, second, joint, targets):
    arrays = [np.asarray(p, dtype=np.float64) for p in (first, second, joint)]
    y = np.asarray(targets)
    shape = arrays[0].shape
    if len(shape) != 2 or shape[0] == 0 or shape[1] < 2:
        raise ValueError("nonempty classification probabilities required")
    if y.shape != (shape[0],) or y.dtype.kind not in "iu" or (y < 0).any() or (y >= shape[1]).any():
        raise ValueError("aligned integer targets required")
    for p in arrays:
        if (
            p.shape != shape
            or not np.isfinite(p).all()
            or (p < 0).any()
            or (p > 1).any()
            or not np.allclose(p.sum(1), 1, atol=1e-6, rtol=0)
        ):
            raise ValueError("aligned normalized probabilities required")
    correct = [p.argmax(1) == y for p in arrays]
    a, b, j = correct
    losses = [-np.log(np.maximum(p[np.arange(len(y)), y], 1e-30)) for p in arrays]
    la, lb, lj = losses
    return dict(
        count=len(y),
        first_correct=int(a.sum()),
        second_correct=int(b.sum()),
        joint_correct=int(j.sum()),
        both_correct=int((a & b).sum()),
        first_only_correct=int((a & ~b).sum()),
        second_only_correct=int((~a & b).sum()),
        neither_correct=int((~a & ~b).sum()),
        joint_loses_first_correct=int((a & ~j).sum()),
        joint_rescues_first_wrong=int((~a & j).sum()),
        joint_loses_both_correct=int((a & b & ~j).sum()),
        selection_oracle_accuracy=float((a | b).mean()),
        first_nll=float(la.mean()),
        second_nll=float(lb.mean()),
        joint_nll=float(lj.mean()),
        second_lower_loss_count=int((lb < la).sum()),
        joint_lower_loss_than_first_count=int((lj < la).sum()),
        label_aware_selection_oracle_nll=float(np.minimum(la, lb).mean()),
        joint_minus_first_nll=float((lj - la).mean()),
    )
