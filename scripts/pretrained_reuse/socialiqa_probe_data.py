"""Dependency-light SocialIQA input validation and group splitting."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ANSWERS = ("answerA", "answerB", "answerC")


def normalized(value: str) -> str:
    return " ".join(value.casefold().split())


def read_split(data_dir: Path, name: str) -> list[dict[str, Any]]:
    items = [
        json.loads(line)
        for line in (data_dir / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    labels = (data_dir / f"{name}-labels.lst").read_text(encoding="utf-8").splitlines()
    if len(items) != len(labels):
        raise ValueError(f"{name}: {len(items)} items but {len(labels)} labels")
    result = []
    for index, (item, label) in enumerate(zip(items, labels, strict=True)):
        if set(ANSWERS) - item.keys() or not {"context", "question"} <= item.keys():
            raise ValueError(f"{name}:{index}: missing required SocialIQA fields")
        if label not in {"1", "2", "3"}:
            raise ValueError(f"{name}:{index}: invalid 1-based label {label!r}")
        result.append(
            {
                **item,
                "id": f"{name}-{index:05d}",
                "target": int(label) - 1,
                "group": normalized(item["context"]),
            }
        )
    return result


def grouped_split(
    items: list[dict[str, Any]], seed: int, validation_fraction: float
) -> tuple[list[int], list[int]]:
    groups: dict[str, list[int]] = {}
    for index, item in enumerate(items):
        groups.setdefault(item["group"], []).append(index)
    ordered = sorted(groups, key=lambda key: hashlib.sha256(f"{seed}:{key}".encode()).digest())
    target = max(1, round(len(items) * validation_fraction))
    validation_groups: set[str] = set()
    selected = 0
    for group in ordered:
        if selected >= target:
            break
        validation_groups.add(group)
        selected += len(groups[group])
    validation = [i for i, item in enumerate(items) if item["group"] in validation_groups]
    train = [i for i, item in enumerate(items) if item["group"] not in validation_groups]
    if not train or not validation:
        raise ValueError("group split produced an empty partition")
    if {items[i]["group"] for i in train} & {items[i]["group"] for i in validation}:
        raise AssertionError("context group leaked across train/validation")
    return train, validation
