import importlib

import pytest


def _diagnostic():
    pytest.importorskip("torch")
    return importlib.import_module("scripts.diagnose_audio_multiquery_audio_dependence")


def test_audio_shuffle_is_a_deterministic_derangement():
    diagnostic = _diagnostic()
    first = diagnostic._derangement(32, seed=17)
    second = diagnostic._derangement(32, seed=17)

    assert diagnostic.torch.equal(first, second)
    assert sorted(first.tolist()) == list(range(32))
    assert diagnostic.torch.all(first != diagnostic.torch.arange(32))


def test_audio_shuffle_requires_multiple_observations():
    diagnostic = _diagnostic()
    try:
        diagnostic._derangement(1, seed=17)
    except ValueError as error:
        assert "at least two" in str(error)
    else:
        raise AssertionError("single observation should not be shuffleable")


def test_speaker_bootstrap_keeps_each_sampled_speaker_cluster_whole():
    diagnostic = _diagnostic()
    rows = [
        {"speaker_group_sha256": "a", "id": "a1"},
        {"speaker_group_sha256": "a", "id": "a2"},
        {"speaker_group_sha256": "b", "id": "b1"},
    ]

    draws = diagnostic._group_bootstrap_indices(rows, replicates=8, seed=4)

    assert len(draws) == 8
    for draw in draws:
        counts = diagnostic.torch.bincount(
            diagnostic.torch.tensor(draw), minlength=3
        ).tolist()
        assert counts[0] == counts[1]
        assert counts[2] in (0, 1, 2)


def test_macro_metrics_are_unweighted_over_the_four_query_types():
    diagnostic = _diagnostic()
    per_task = {
        task: {
            "accuracy": index,
            "nll": index + 1,
            "brier": index + 2,
            "ece_15_bins": index + 3,
        }
        for index, task in enumerate(diagnostic.QUESTION_TASKS)
    }

    macro = diagnostic._macro_metrics(per_task)

    assert macro == {"accuracy": 1.5, "nll": 2.5, "brier": 3.5, "ece_15_bins": 4.5}


def test_metric_values_use_multiclass_brier_and_calibration():
    diagnostic = _diagnostic()
    logits = diagnostic.torch.tensor([[4.0, 0.0], [0.0, 4.0]], dtype=diagnostic.torch.float64)
    targets = diagnostic.torch.tensor([0, 0])

    metrics = diagnostic._metric_values(logits, targets)

    assert metrics["accuracy"] == 0.5
    assert metrics["nll"] > 2.0
    assert metrics["brier"] > 0.5
    assert 0.0 <= metrics["ece_15_bins"] <= 1.0
