import copy
from collections import Counter

import pytest

from scripts.binding_replay_coverage import (
    cache_entry,
    checked_logits,
    mixed_replay,
    verify_intervention,
    verify_split_integrity,
)
from scripts.condition_binding_corpus import schedule


def data():
    original = [
        dict(
            id=f"o:{i}",
            group=f"o:{i}",
            split="train",
            source="original",
            state=str(i),
            task="choice",
            options=["a", "b"],
            target="a",
        )
        for i in range(2048)
    ]
    tasks = ["extremum"] * 4 + ["temporal_order"] * 2 + ["count", "conditional_selection"]
    curriculum = [
        dict(
            id=f"c:{i}:{j}",
            group=f"c:{i}",
            split="train",
            state=str(i),
            task=task,
            options=["x", "y"],
            target="x",
        )
        for i in range(160)
        for j, task in enumerate(tasks)
    ]
    return original, curriculum


def test_mixed_replay_keeps_budget_order_and_complete_curriculum_states():
    original, curriculum = data()
    order = schedule(2048, 2048)
    result = mixed_replay(original, curriculum, order)
    assert result == mixed_replay(original, list(reversed(curriculum)), order)
    assert len(result) == len({r["id"] for r in result}) == 2048
    assert Counter(r["source"] for r in result) == {"original": 1024, "curriculum": 1024}
    for batch in order:
        indices = [i for source, i in batch if source == 1]
        assert [result[i]["source"] for i in indices] == ["original"] * 2 + ["curriculum"] * 2
        assert [result[i] for i in indices[:2]] == [original[i] for i in indices[:2]]
    selected = [r for r in result if r["source"] == "curriculum"]
    assert set(Counter(r["group"] for r in selected).values()) == {8}
    assert len({r["group"] for r in selected}) == 128
    assert Counter(r["task"] for r in selected) == dict(
        extremum=512, temporal_order=256, count=128, conditional_selection=128
    )
    by_id = {r["id"]: r for r in curriculum}
    assert all({k: v for k, v in r.items() if k != "source"} == by_id[r["id"]] for r in selected)


def test_replay_rejects_nontrain_duplicates_incomplete_groups_or_schedule():
    original, curriculum = data()
    order = schedule(2048, 2048)
    for bad in (
        curriculum[:-1],
        curriculum + [curriculum[0]],
        [dict(r, split="validation") for r in curriculum],
    ):
        with pytest.raises(ValueError):
            mixed_replay(original, bad, order)
    bad_order = copy.deepcopy(order)
    bad_order[1] = bad_order[0]
    with pytest.raises(ValueError):
        mixed_replay(original, curriculum, bad_order)
    with pytest.raises(ValueError):
        mixed_replay(original[:-1], curriculum, order)


def test_cache_is_train_only_and_binds_source_question_and_option_order():
    original, curriculum = data()
    for row in (original[0], dict(curriculum[0], source="curriculum")):
        cached = cache_entry(row, [0.1, -0.5])
        assert checked_logits(row, cached) == [0.1, -0.5]
        for key, value in [
            ("split", "validation"),
            ("source", "binding"),
            ("options", ["b", "a"]),
            ("question", "changed"),
        ]:
            with pytest.raises(ValueError):
                checked_logits(dict(row, **{key: value}), cached)
    with pytest.raises(ValueError):
        cache_entry(original[0], [float("nan"), 1])


def test_only_declared_intervention_config_changes_are_allowed():
    from scripts.evaluate_semantic_state_bridge import read

    parent = read("configs/pretrained_reuse/binding_language_v1.json")
    current = read("configs/pretrained_reuse/binding_replay_coverage_v1.json")
    verify_intervention(current, parent)
    for key, value in [
        ("steps", 1024),
        ("learning_rate", 1e-4),
        ("retention_kl_coefficient", 2),
        ("language_policy", "changed"),
    ]:
        with pytest.raises(ValueError):
            verify_intervention(dict(current, **{key: value}), parent)


def test_split_gate_rejects_group_overlap_even_if_text_or_row_id_differs():
    train = [dict(id="a", group="scene1", state='{"x":1}')]
    validation = [dict(id="b", group="scene2", state='{"x":2}')]
    verify_split_integrity(train, validation)
    with pytest.raises(ValueError, match="group overlap"):
        verify_split_integrity(train, [dict(validation[0], group="scene1")])
    with pytest.raises(ValueError, match="state overlap"):
        verify_split_integrity(train, [dict(validation[0], state=' { "x": 1 } ')])
