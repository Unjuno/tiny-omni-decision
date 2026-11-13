import json

import numpy as np
import pytest

from scripts import condition_binding_corpus as corpus
from scripts.semantic_state_diagnostic import FIELDS, QUESTIONS


def test_partition_excludes_every_seen_and_probe_fact_combination():
    states = corpus.all_states()
    known = states[:1000]
    probe = known[:20] + states[1000:1020]
    partitions = corpus.partition(
        known, probe, train_count=64, validation_count=32, transfer_count=32
    )
    assert partitions == corpus.partition(
        list(reversed(known)), probe, train_count=64, validation_count=32, transfer_count=32
    )
    sets = {k: {corpus.identity(x) for x in v} for k, v in partitions.items()}
    assert not sets["train"] & {corpus.identity(s) for s in probe}
    for name in ("validation", "transfer"):
        assert not sets[name] & {corpus.identity(s) for s in known + probe}
    assert not sets["train"] & sets["validation"]
    assert not sets["validation"] & sets["transfer"]
    with pytest.raises(ValueError):
        corpus.partition(known[:3], probe, train_count=64)


def test_questions_keep_all_choices_and_targets_after_order_reversal():
    state = dict(color="red", texture="star", count="5", shape="star")
    for qnovel in (False, True):
        for onovel in (False, True):
            rows = corpus.questions_for(state, "transfer", qnovel, onovel)
            assert len(rows) == 4
            assert len({r["target_id"] for r in rows}) == 4
            assert len({r["state"] for r in rows}) == 1
            for i, row in enumerate(rows):
                assert len(row["options"]) == len(set(row["option_ids"])) == 40
                assert row["target_id"] == f"{FIELDS[i]}:{state[FIELDS[i]]}"
                target = row["option_ids"].index(row["target_id"])
                reversed_row = corpus.reverse_options(row)
                assert reversed_row["option_ids"].index(row["target_id"]) == 39 - target
                assert row["question"] != QUESTIONS[i] if qnovel else True
                assert json.loads(row["state"]) == state
                assert "target" not in row["state"] and "group" not in row["state"]


def test_training_and_novel_templates_do_not_overlap():
    train = {q for templates in corpus.TRAIN_QUESTIONS for q in templates}
    assert not train & set(corpus.NOVEL_QUESTIONS)
    state = corpus.all_states()[0]
    old = corpus.questions_for(state, "train")
    new = corpus.questions_for(state, "transfer", True, True)
    for a, b in zip(old, new, strict=True):
        assert not set(a["options"]) & set(b["options"])
        assert a["option_ids"] == b["option_ids"]


def test_retention_gates_use_every_task_and_calibration_not_only_aggregate():
    base = {
        "original": {"accuracy": 0.8, "nll": 0.4, "brier": 0.2, "ece": 0.05},
        "curriculum": {"accuracy": 0.9, "nll": 0.3, "brier": 0.1, "ece": 0.02},
    }
    tasks = {"choice": 0.8, "count": 0.6}
    assert corpus.retention_ok(base, tasks, base, tasks)
    assert not corpus.retention_ok(base, dict(tasks, count=0.58), base, tasks)
    degraded = {**base, "original": dict(base["original"], nll=0.43)}
    assert not corpus.retention_ok(degraded, tasks, base, tasks)
    assert not corpus.retention_ok(base, {"choice": 0.9}, base, tasks)
    assert not corpus.retention_ok(base, dict(tasks, count=float("nan")), base, tasks)


def test_selector_rejects_retention_failures_and_preserves_earliest_tie():
    history = [
        {"step": 0, "eligible": True, "binding_nll": 2.0},
        {"step": 128, "eligible": False, "binding_nll": 0.1},
        {"step": 256, "eligible": True, "binding_nll": 1.0},
        {"step": 384, "eligible": True, "binding_nll": 1.0},
        {"step": 512, "eligible": True, "binding_nll": float("nan")},
    ]
    assert corpus.select_checkpoint(history) == 256
    assert corpus.select_checkpoint(history[:2]) == 0
    with pytest.raises(ValueError):
        corpus.select_checkpoint([history[1]])


def test_source_schedule_uses_every_row_once_and_is_deterministic():
    first = corpus.schedule(2048, 2048)
    assert first == corpus.schedule(2048, 2048)
    assert len(first) == 512
    for source in range(2):
        flat = [i for step in first for s, i in step if s == source]
        assert len(flat) == len(set(flat)) == 2048
        assert set(flat) == set(range(2048))
    assert all([s for s, _ in step] == [0] * 4 + [1] * 4 for step in first)
    with pytest.raises(ValueError):
        corpus.schedule(2047, 2048)


def test_pair_metrics_keep_state_grouping():
    target = np.tile(np.array([0, 10, 20, 30]), (3, 1))
    predicted = target.copy()
    predicted[0, 3] = 0
    result = corpus.pair_metrics(predicted, target)
    assert result["all_four_correct"] == pytest.approx(2 / 3)
    assert result["pair_both_correct"] == pytest.approx(15 / 18)


def test_padded_ragged_scores_preserve_true_option_probabilities():
    from scripts.condition_binding_training import pack_scores
    from scripts.semantic_state_diagnostic import metrics

    values = [np.log([0.8, 0.2]), np.log([0.1, 0.2, 0.7])]
    packed = pack_scores(values)
    assert packed.shape == (2, 3)
    result = metrics(packed, np.array([0, 2]))
    assert result["accuracy"] == 1
    assert result["nll"] == pytest.approx(-np.log(0.56) / 2)
    assert result["brier"] == pytest.approx((0.08 + 0.14) / 2)
