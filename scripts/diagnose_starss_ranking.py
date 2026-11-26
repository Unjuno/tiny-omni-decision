"""Threshold-free diagnosis of saved development predictions; no model fitting."""

import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.probe_starss23_windows import EVENTS


def ranking_metrics(targets, scores):
    y, p = np.asarray(targets), np.asarray(scores)
    if y.ndim != 1 or y.shape != p.shape or not y.size:
        raise ValueError("nonempty matching vectors required")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all():
        raise ValueError("binary targets and finite scores required")
    positives, negatives = int(y.sum()), int((y == 0).sum())
    auc = None
    if positives and negatives:
        differences = p[y == 1][:, None] - p[y == 0][None, :]
        auc = float(np.mean((differences > 0) + 0.5 * (differences == 0)))
    ap = None
    if positives:
        order = np.argsort(-p, kind="stable")
        sorted_y, sorted_p = y[order], p[order]
        # Include each complete equal-score group; never break ties using labels.
        ends = np.r_[np.flatnonzero(sorted_p[:-1] != sorted_p[1:]), len(y) - 1]
        tp = np.cumsum(sorted_y)[ends]
        recall = tp / positives
        precision = tp / (ends + 1)
        ap = float(np.sum(np.diff(np.r_[0.0, recall]) * precision))
    return {"positive": positives, "negative": negatives,
            "prevalence": positives / len(y), "auroc": auc, "average_precision": ap}


def main():
    root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    output = root / "starss-ranking-diagnosis-01"
    output.mkdir(exist_ok=False)
    result = {"selection_or_threshold_tuning": False, "blind_audit": False,
              "conditions": {}, "input_hashes": {}}
    for arm in ("pooled", "pretrained_logits"):
        path = root / "starss-pretrained-head-comparison-01" / arm / "validation_predictions.npz"
        result["input_hashes"][arm] = hashlib.sha256(path.read_bytes()).hexdigest()
        data = np.load(path, allow_pickle=False)
        result["conditions"][arm] = {
            event: ranking_metrics(data["targets"][:, i], data["positive_probabilities"][:, i])
            for i, event in enumerate(EVENTS)
        }
    result["limitation"] = "Repeated development data, selected checkpoints, two rooms."
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["conditions"]["pooled"], indent=2))


if __name__ == "__main__":
    main()
