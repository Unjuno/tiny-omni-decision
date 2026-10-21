from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from scripts.evaluate_clevrer_fresh_reasoning_holdout import paired_scene_bootstrap


def test_cli_main_runs_after_bootstrap_function_is_defined():
    source = Path(__file__).parents[1] / "scripts/evaluate_clevrer_fresh_reasoning_holdout.py"
    module = ast.parse(source.read_text(encoding="utf-8"))
    bootstrap = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "paired_scene_bootstrap"
    )
    main_guard = next(
        node
        for node in module.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and any(
            isinstance(item, ast.Constant) and item.value == "__main__"
            for item in node.test.comparators
        )
    )
    assert main_guard.lineno > bootstrap.end_lineno


def test_paired_scene_bootstrap_is_reproducible_and_zero_for_identical_predictions():
    rows = [
        {"scene_index": scene, "question_type": kind}
        for scene in range(4)
        for kind in ("explanatory", "predictive", "counterfactual")
    ]
    targets = [index % 2 for index in range(len(rows))]
    probabilities = np.asarray(
        [[0.8, 0.2] if target == 0 else [0.25, 0.75] for target in targets], dtype=np.float64
    )

    first = paired_scene_bootstrap(
        rows,
        targets,
        reference_probabilities=probabilities,
        candidate_probabilities=probabilities.copy(),
        replicates=100,
        seed=17,
    )
    replay = paired_scene_bootstrap(
        rows,
        targets,
        reference_probabilities=probabilities,
        candidate_probabilities=probabilities.copy(),
        replicates=100,
        seed=17,
    )

    assert first == replay
    assert first["independent_scenes"] == 4
    assert first["replicates"] == 100
    for metric in first["macro_question_type_difference"]:
        assert first["macro_question_type_difference"][metric] == pytest.approx(0.0)
        assert first["difference_ci_95"][metric] == [0.0, 0.0]
