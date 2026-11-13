import numpy as np
import pytest

from scripts.analyze_native_temporal_fit import bootstrap, per_example


def test_comparison_per_example_uses_all_supplied_candidates():
    rows = [dict(target_index=1), dict(target_index=0)]
    logits = np.full((2, 40), -1e30, dtype=np.float32)
    logits[0, :2] = [0, np.log(3)]
    logits[1, :3] = [0, 0, 0]
    result = per_example(rows, logits)
    assert result["accuracy"].tolist() == [1, 1]
    assert result["nll"][0] == pytest.approx(-np.log(0.75))
    assert result["brier"][0] == pytest.approx(0.125)
    assert result["brier"][1] == pytest.approx(2 / 3)


def test_cluster_bootstrap_is_paired_deterministic_and_zero_for_identity():
    rows = [
        dict(modality="video", task=task, group=str(g))
        for g in range(3)
        for task in ("static_descriptive", "temporal_descriptive")
    ]
    values = {k: np.arange(6, dtype=float) for k in ("accuracy", "nll", "brier")}
    result = bootstrap(rows, values, values)
    assert result == bootstrap(rows, values, values)
    for part in result.values():
        assert part["groups"] == 3
        for key in values:
            assert part[key] == dict(lstm_minus_baseline=0.0, low=0.0, high=0.0)


def test_cluster_bootstrap_point_estimate_weights_questions_not_scene_means():
    rows = [
        dict(modality="video", task=task, group=g)
        for task, g in [
            ("static_descriptive", "a"),
            ("temporal_descriptive", "a"),
            ("temporal_descriptive", "b"),
        ]
    ]
    a = {"accuracy": np.zeros(3)}
    b = {"accuracy": np.array([0.0, 0.0, 1.0])}
    result = bootstrap(rows, a, b)
    assert result["all"]["accuracy"]["lstm_minus_baseline"] == pytest.approx(1 / 3)
    assert result["temporal_descriptive"]["accuracy"]["lstm_minus_baseline"] == 0.5
    assert result["all"]["accuracy"]["low"] == 0
    assert result["all"]["accuracy"]["high"] == 1
