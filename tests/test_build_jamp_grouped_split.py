from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.pretrained_reuse.build_jamp_grouped_split import build_split


def _rows() -> list[dict[str, str]]:
    rows = []
    row_id = 1
    for template in range(10):
        for label in ("contradiction", "entailment", "neutral"):
            for example in range(5):
                rows.append(
                    {
                        "num": str(row_id),
                        "premise": f"premise {template} {label} {example}",
                        "hypothesis": f"hypothesis {template} {label} {example}",
                        "gold_label": label,
                        "template_num": str(template),
                    }
                )
                row_id += 1
    return rows


def test_jamp_template_groups_are_disjoint_and_split_is_deterministic() -> None:
    rows = _rows()
    train_a, validation_a, assignment_a = build_split(rows, seed=17)
    train_b, validation_b, assignment_b = build_split(rows, seed=17)

    assert assignment_a == assignment_b
    assert [row["num"] for row in train_a] == [row["num"] for row in train_b]
    assert [row["num"] for row in validation_a] == [row["num"] for row in validation_b]
    assert {row["template_num"] for row in train_a}.isdisjoint(
        row["template_num"] for row in validation_a
    )
    assert {row["num"] for row in train_a}.isdisjoint(row["num"] for row in validation_a)
    assert len(train_a) + len(validation_a) == len(rows)


def test_jamp_split_rejects_normalized_content_overlap() -> None:
    rows = _rows()
    rows[1]["premise"] = "　" + rows[0]["premise"]
    rows[1]["hypothesis"] = rows[0]["hypothesis"]

    with pytest.raises(ValueError, match="Normalized premise/hypothesis pair repeats"):
        build_split(rows, seed=17)
