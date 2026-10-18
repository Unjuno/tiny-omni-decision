from collections import Counter

import pytest


def test_long_schedule_has_learning_rate_after_the_old_horizon():
    from scripts.native_long_budget import EVALUATION_STEPS, MAX_STEPS, learning_rate

    assert MAX_STEPS == 32768
    assert EVALUATION_STEPS == (0, 512, 1024, 2048, 4096, 8192, 12288, 16384, 24576, 32768)
    assert learning_rate(1) == pytest.approx(0.001 / 62)
    assert learning_rate(62) == 0.001
    assert learning_rate(2048) > 0.00099
    assert learning_rate(16384) > 0.0005
    assert learning_rate(MAX_STEPS) == 0
    assert all(learning_rate(i + 1) <= learning_rate(i) for i in range(62, MAX_STEPS))


@pytest.mark.parametrize("step", [True, 0, -1, 32769, 1.5])
def test_invalid_update_rejected(step):
    from scripts.native_long_budget import learning_rate

    with pytest.raises(ValueError):
        learning_rate(step)


def test_repeated_plan_preserves_prefix_counts_and_global_occurrences():
    from scripts.native_long_budget import extend_plan

    pool = [9, 8, 9, 7, 6, 5, 4, 3] * 2
    batches = [list(range(8)), list(range(8, 16))]
    plan = extend_plan(pool, batches, 6)
    assert plan == extend_plan(pool, batches, 6)
    assert plan["order"] == [[pool[o] for o in b] for b in batches] * 3
    assert plan["occurrences"][:2] == batches
    assert sorted(o for b in plan["occurrences"] for o in b) == list(range(48))
    assert Counter(i for b in plan["order"] for i in b) == Counter(pool * 3)
    assert plan["unique_row_indices"] == sorted(set(pool))
    assert plan["cycles"] == 3


@pytest.mark.parametrize("case", ["partial", "duplicate", "missing", "width", "negative", "empty"])
def test_invalid_plan_fails_closed(case):
    from scripts.native_long_budget import extend_plan

    pool, batches, steps = list(range(16)), [list(range(8)), list(range(8, 16))], 6
    if case == "partial":
        steps = 5
    elif case == "duplicate":
        batches[1][-1] = 0
    elif case == "missing":
        pool.append(16)
    elif case == "width":
        batches[0].pop()
    elif case == "negative":
        pool[0] = -1
    else:
        batches = []
    with pytest.raises(ValueError):
        extend_plan(pool, batches, steps)


def test_long_curve_bootstrap_nll_stays_finite_for_confident_wrong_answers():
    import numpy as np

    from scripts.analyze_native_long_budget import per_example
    from scripts.semantic_state_diagnostic import metrics

    scores = np.full((2, 40), -1e30, dtype=np.float32)
    scores[0, :2], scores[1, :2] = [0, -1000], [2, 1]
    target = np.array([1, 0])
    values = per_example([dict(target_index=int(t)) for t in target], scores)
    expected = metrics(scores, target)
    for name, vector in values.items():
        assert np.isfinite(vector).all()
        assert float(vector.mean()) == pytest.approx(expected[name], abs=1e-12)


def test_unique_coverage_policies_have_fixed_probe_sizes():
    from scripts.native_long_budget import unique_probe_row_count

    assert unique_probe_row_count("repeat") == 6656
    assert unique_probe_row_count("fresh") == 12800
    with pytest.raises(ValueError):
        unique_probe_row_count("unknown")


def test_long_coverage_comparison_delta_is_fresh_minus_repeat_per_group():
    from scripts.analyze_native_unique_coverage_long import summarize_deltas, unique_row_summary

    control = {
        "macro": {key: 1.0 for key in ("accuracy", "nll", "brier", "ece")},
        **{
            group: {"video": {key: 1.0 for key in ("accuracy", "nll", "brier", "ece")}}
            for group in ("modalities", "sources", "tasks")
        },
    }
    fresh = {
        "macro": {key: 1.25 for key in ("accuracy", "nll", "brier", "ece")},
        **{
            group: {"video": {key: 0.75 for key in ("accuracy", "nll", "brier", "ece")}}
            for group in ("modalities", "sources", "tasks")
        },
    }

    delta = summarize_deltas(control, fresh)

    assert delta["macro"]["accuracy"] == pytest.approx(0.25)
    assert delta["modalities"]["video"]["nll"] == pytest.approx(-0.25)
    assert delta["sources"]["video"]["brier"] == pytest.approx(-0.25)
    assert delta["tasks"]["video"]["ece"] == pytest.approx(-0.25)

    summary = unique_row_summary(
        [
            {"id": "text-1", "modality": "text", "source": "synthetic", "task": "rank"},
            {"id": "image-1", "modality": "image", "source": "clevr4", "task": "count"},
        ],
        ["text-1", "image-1"],
    )
    assert summary == {
        "total": 2,
        "by_modality": {"image": 1, "text": 1},
        "by_source": {"clevr4": 1, "synthetic": 1},
        "by_source_task": {"clevr4:count": 1, "synthetic:rank": 1},
    }
    with pytest.raises(ValueError, match="unique TRAIN"):
        unique_row_summary([], ["missing"])
