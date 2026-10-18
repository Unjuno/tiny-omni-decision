import importlib

import pytest

np = pytest.importorskip("numpy")
analysis = importlib.import_module("scripts.analyze_av_event_readout_comparison")


def _prediction(row_id, video_id, target, probabilities):
    return {
        "id": row_id,
        "video_id": video_id,
        "target_index": target,
        "candidate_count": len(probabilities),
        "candidate_order_sha256": "c" * 64,
        "probabilities": probabilities,
    }


def test_metrics_support_variable_option_counts_and_uniform_baseline():
    rows = [
        _prediction("e1", "v1", 0, [0.8, 0.2]),
        _prediction("e2", "v2", 2, [1 / 3, 1 / 3, 1 / 3]),
    ]
    assert analysis.score_rows(rows)["accuracy"] == 0.5
    uniform = analysis.uniform_predictions(rows)
    assert uniform[0]["probabilities"] == [0.5, 0.5]
    assert analysis.score_rows(uniform)["n"] == 2


def test_video_cluster_bootstrap_is_deterministic_and_requires_matched_rows():
    left = [
        _prediction("e1", "v1", 0, [0.9, 0.1]),
        _prediction("e2", "v2", 1, [0.9, 0.1]),
    ]
    right = [
        _prediction("e1", "v1", 0, [0.5, 0.5]),
        _prediction("e2", "v2", 1, [0.5, 0.5]),
    ]
    first = analysis.paired_video_bootstrap(left, right, draws=50, seed=17)
    second = analysis.paired_video_bootstrap(left, right, draws=50, seed=17)
    assert first == second
    with pytest.raises(ValueError, match="do not match"):
        analysis.paired_video_bootstrap(left, right[:-1], draws=5)
