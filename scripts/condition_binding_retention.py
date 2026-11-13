"""TRAIN-only parent option distributions and equal-question retention KL."""

import hashlib
import json

import numpy as np


def question_hash(row):
    if row.get("split") != "train" or row.get("source") != "original":
        raise ValueError("retention targets require original TRAIN rows")
    return hashlib.sha256(
        json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def cache_entry(row, logits):
    values = np.asarray(logits)
    if values.shape != (len(row["options"]),) or not np.isfinite(values).all():
        raise ValueError("finite per-candidate logits required")
    return dict(question_sha256=question_hash(row), logits=values.tolist())


def checked_logits(row, cached):
    if cached["question_sha256"] != question_hash(row):
        raise ValueError("retention question or candidate order changed")
    return cache_entry(row, cached["logits"])["logits"]


def retention_kl(scores, widths, teacher_scores):
    import torch
    from torch.nn import functional as functional

    if (
        not widths
        or min(widths) < 2
        or sum(widths) != scores.numel()
        or scores.ndim != 1
        or teacher_scores.shape != scores.shape
        or not torch.isfinite(scores).all()
        or not torch.isfinite(teacher_scores).all()
    ):
        raise ValueError("finite aligned variable-choice scores required")
    teacher_scores = teacher_scores.detach()
    return torch.stack(
        [
            (
                teacher.log_softmax(0).exp()
                * (teacher.log_softmax(0) - functional.log_softmax(student, dim=0))
            ).sum()
            for student, teacher in zip(
                scores.split(widths), teacher_scores.split(widths), strict=True
            )
        ]
    ).mean()
