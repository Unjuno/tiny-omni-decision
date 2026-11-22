"""Shared JCoLA corpus checks and supplied-option scoring primitives."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

OPTIONS = ("不適格（文法的に許容されない）", "適格（文法的に許容される）")
QUESTION = "次の文について、文法上の許容性を選んでください。"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_sentence(text: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text)).casefold()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def validate_corpus(rows: list[dict[str, Any]], name: str) -> None:
    ids = [str(row["uid"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{name} has duplicate uids")
    for row in rows:
        if row.get("label") not in (0, 1):
            raise ValueError(f"{name} has a non-binary label")
        if not str(row.get("sentence", "")).strip() or not str(row.get("source", "")):
            raise ValueError(f"{name} is missing sentence/source")


def assert_disjoint_groups_and_content(
    train: list[dict[str, Any]], validation: list[dict[str, Any]], name: str
) -> None:
    train_sources = {str(row["source"]) for row in train}
    validation_sources = {str(row["source"]) for row in validation}
    if train_sources & validation_sources:
        raise ValueError(f"{name} source groups cross partitions")
    train_text = {normalize_sentence(str(row["sentence"])) for row in train}
    validation_text = {normalize_sentence(str(row["sentence"])) for row in validation}
    if train_text & validation_text:
        raise ValueError(f"{name} normalized sentence content crosses partitions")


class CandidateScorer(nn.Module):
    def __init__(self, hidden_size: int, readout_size: int = 128) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_size, readout_size),
            nn.Tanh(),
            nn.Linear(readout_size, 1),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.layers(features).squeeze(-1)


def metrics(logits: Tensor, targets: Tensor) -> dict[str, float | int]:
    probabilities = torch.softmax(logits, dim=-1)
    predictions = probabilities.argmax(-1)
    targets = targets.long()
    correct = predictions.eq(targets)
    recalls = [float(correct[targets == label].float().mean()) for label in (0, 1)]
    confidence = probabilities.max(-1).values
    ece = 0.0
    for index in range(15):
        lo, hi = index / 15, (index + 1) / 15
        selected = (confidence >= lo) & ((confidence < hi) if index < 14 else (confidence <= hi))
        if selected.any():
            ece += float(selected.float().mean()) * abs(
                float(confidence[selected].mean()) - float(correct[selected].float().mean())
            )
    tp = int(((predictions == 1) & (targets == 1)).sum())
    tn = int(((predictions == 0) & (targets == 0)).sum())
    fp = int(((predictions == 1) & (targets == 0)).sum())
    fn = int(((predictions == 0) & (targets == 1)).sum())
    denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    return {
        "count": int(targets.numel()),
        "accuracy": float(correct.float().mean()),
        "macro_accuracy": sum(recalls) / 2,
        "unacceptable_recall": recalls[0],
        "acceptable_recall": recalls[1],
        "nll": float(F.cross_entropy(logits, targets)),
        "brier": float((probabilities - F.one_hot(targets, 2)).square().sum(-1).mean()),
        "ece_15": ece,
        "mcc": (tp * tn - fp * fn) / denominator if denominator else 0.0,
        "mean_confidence": float(confidence.mean()),
    }
