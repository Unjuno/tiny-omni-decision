from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from scripts.evaluate_clevrer_fresh_reasoning_holdout import paired_scene_bootstrap


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
