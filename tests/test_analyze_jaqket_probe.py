from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.pretrained_reuse.analyze_jaqket_probe import CONDITIONS, analyze


def test_grouped_bootstrap_is_deterministic_and_checks_prediction_order(tmp_path: Path) -> None:
    validation = tmp_path / "validation.jsonl"
    rows = []
    for index in range(4):
        options = [f"option-{index}-{choice}" for choice in range(20)]
        options[0] = f"answer-{index // 2}"
        rows.append(
            {
                "qid": f"q-{index}",
                "question": f"question {index}",
                "answer_entity": f"answer-{index // 2}",
                "answer_candidates": options,
            }
        )
    validation.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    probabilities = np.full((20,), 0.05)
    for directory in CONDITIONS.values():
        output = tmp_path / directory
        output.mkdir()
        predictions = []
        for row in rows:
            row_probabilities = probabilities.copy()
            row_probabilities[:] = 0.9 / 19
            row_probabilities[0] = 0.1
            predictions.append(
                {
                    "qid": row["qid"],
                    "probabilities": row_probabilities.tolist(),
                }
            )
        (output / "validation-predictions.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in predictions), encoding="utf-8"
        )

    first = analyze(validation, tmp_path, repetitions=50, seed=1729)
    second = analyze(validation, tmp_path, repetitions=50, seed=1729)

    assert first == second
    assert first["answer_entity_groups"] == 2
    assert first["metrics"]["bert-base-pair"]["accuracy"] == 1.0
    assert "bert-base-pair-minus-distil4-pair" in first["paired_differences"]
