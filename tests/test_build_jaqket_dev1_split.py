from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.pretrained_reuse.build_jaqket_dev1_split import create_split


def _row(qid: str, question: str, target: str, offset: int) -> dict[str, object]:
    options = [f"choice-{offset}-{index}" for index in range(20)]
    options[0] = target
    return {
        "qid": qid,
        "question": question,
        "answer_entity": target,
        "answer_candidates": options,
    }


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )


def test_split_is_deterministic_and_keeps_duplicate_answer_groups_together(tmp_path: Path) -> None:
    rows = [
        _row(f"id-{index}", f"question {index}", f"answer-{index}", index) for index in range(10)
    ]
    rows[1]["answer_entity"] = rows[0]["answer_entity"]
    rows[1]["answer_candidates"][0] = rows[0]["answer_entity"]
    source = tmp_path / "dev1.jsonl"
    _write_rows(source, rows)

    first = create_split(source, tmp_path / "split-a")
    second = create_split(source, tmp_path / "split-b")

    assert first["split"] == second["split"]
    assert first["source"] == second["source"]
    assert first["split"]["group_components"] == 9
    assert first["split"]["cross_split_overlap"] == {
        "qid": 0,
        "normalized_question": 0,
        "answer_entity": 0,
        "unordered_candidate_set": 0,
    }
    train = [
        json.loads(line) for line in (tmp_path / "split-a" / "train.jsonl").read_text().splitlines()
    ]
    validation = [
        json.loads(line)
        for line in (tmp_path / "split-a" / "validation.jsonl").read_text().splitlines()
    ]
    train_ids = {row["qid"] for row in train}
    validation_ids = {row["qid"] for row in validation}
    assert ("id-0" in train_ids) == ("id-1" in train_ids)
    assert ("id-0" in validation_ids) == ("id-1" in validation_ids)


def test_split_rejects_rows_without_exactly_one_target_option(tmp_path: Path) -> None:
    row = _row("id-0", "question", "answer", 0)
    row["answer_candidates"][0] = "wrong-answer"
    source = tmp_path / "invalid.jsonl"
    _write_rows(source, [row])

    with pytest.raises(ValueError, match="target must occur exactly once"):
        create_split(source, tmp_path / "split")


def test_split_rejects_duplicate_candidate_titles_after_normalization(tmp_path: Path) -> None:
    row = _row("id-0", "question", "answer", 0)
    row["answer_candidates"][1] = row["answer_candidates"][2]
    source = tmp_path / "invalid.jsonl"
    _write_rows(source, [row])

    with pytest.raises(ValueError, match="duplicate normalized candidates"):
        create_split(source, tmp_path / "split")
