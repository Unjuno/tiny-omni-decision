from __future__ import annotations

import json

import pytest

from scripts.pretrained_reuse.socialiqa_probe_data import grouped_split, read_split


def test_grouped_split_is_deterministic_and_keeps_contexts_together() -> None:
    items = [{"group": f"context-{index // 2}", "target": index % 3} for index in range(30)]
    train_a, validation_a = grouped_split(items, seed=17, validation_fraction=0.2)
    train_b, validation_b = grouped_split(items, seed=17, validation_fraction=0.2)
    assert (train_a, validation_a) == (train_b, validation_b)
    assert set(train_a).isdisjoint(validation_a)
    assert set(train_a) | set(validation_a) == set(range(len(items)))
    train_groups = {items[index]["group"] for index in train_a}
    validation_groups = {items[index]["group"] for index in validation_a}
    assert train_groups.isdisjoint(validation_groups)


def test_read_split_aligns_labels_and_normalizes_context_groups(tmp_path) -> None:
    row = {
        "context": "  They   went home. ",
        "question": "Why?",
        "answerA": "A",
        "answerB": "B",
        "answerC": "C",
    }
    (tmp_path / "train.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    (tmp_path / "train-labels.lst").write_text("3\n", encoding="utf-8")
    result = read_split(tmp_path, "train")
    assert result[0]["id"] == "train-00000"
    assert result[0]["target"] == 2
    assert result[0]["group"] == "they went home."


def test_read_split_rejects_misaligned_labels(tmp_path) -> None:
    (tmp_path / "train.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "train-labels.lst").write_text("1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required"):
        read_split(tmp_path, "train")
