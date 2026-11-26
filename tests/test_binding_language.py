import json

import pytest

from scripts.binding_language import diverse_train, heldout_conditions, unused_states
from scripts.condition_binding_corpus import all_states, identity, questions_for


def test_diverse_supervision_changes_only_question_identity_and_language_metadata():
    states = all_states()[::19][:16]
    original = [r for s in states for r in questions_for(s, "train")]
    diverse = diverse_train(original)
    assert len(diverse) == len(original)
    families = {r["language_family"] for r in diverse}
    assert len(families) == 8
    for a, b in zip(original, diverse, strict=True):
        assert b["underlying_id"] == a["id"]
        for key in set(a) - {"question", "id"}:
            assert a[key] == b[key]
        assert b["target_id"] == f"{b['task']}:{json.loads(b['state'])[b['task']]}"
    assert diverse == diverse_train(original)
    for group in range(16):
        block = diverse[group * 4 : group * 4 + 4]
        assert len({r["language_family"] for r in block}) == 1
        assert len({r["target_id"].split(":")[0] for r in block}) == 4


def test_new_holdout_excludes_prior_states_and_instruction_strings():
    blocked = all_states()[:200]
    states = unused_states(blocked, 8)
    assert states == unused_states(blocked[::-1], 8)
    assert not {identity(s) for s in states} & {identity(s) for s in blocked}
    with pytest.raises(ValueError):
        unused_states(all_states(), 1)
    data = heldout_conditions(states)
    original = [r for s in blocked[:16] for r in questions_for(s, "train")]
    training = {r["question"] for r in diverse_train(original)}
    for name, rows in data.items():
        assert len(rows) == 32
        for row in rows:
            assert len(row["options"]) == len(set(row["options"])) == 40
            assert row["target"] == row["options"][row["option_ids"].index(row["target_id"])]
            if name not in ("known", "no_question"):
                assert row["question"] not in training
    for a, b in zip(data["focus_descriptions"], data["reverse_focus_descriptions"], strict=True):
        assert a["options"] == b["options"][::-1]
        assert a["target_id"] == b["target_id"]


def test_language_generation_rejects_incomplete_or_nontrain_groups():
    rows = questions_for(all_states()[0], "train")
    with pytest.raises(ValueError):
        diverse_train(rows[:3])
    with pytest.raises(ValueError):
        diverse_train([dict(r, split="validation") for r in rows])


def test_fixed_512_state_budget_is_balanced_and_does_not_repeat_questions():
    from collections import Counter

    rows = [r for s in all_states()[::19][:512] for r in questions_for(s, "train")]
    transformed = diverse_train(rows)
    assert len(transformed) == len({r["id"] for r in transformed}) == 2048
    assert Counter(r["language_family"] for r in transformed) == dict.fromkeys(range(8), 256)
    assert len({r["group"] for r in transformed}) == 512
    for row in transformed:
        if row["language_family"] == 4:
            assert row["task"] not in row["question"]
            assert all(
                f in row["question"]
                for f in ("color", "texture", "count", "shape")
                if f != row["task"]
            )


def test_quality_requires_both_primary_conditions_and_each_field():
    from scripts.binding_language import CONDITIONS, quality_failures

    good = dict(
        overall=dict(accuracy=0.99, nll=0.05, brier=0.02, ece=0.01),
        tasks={f: dict(accuracy=0.99) for f in ("color", "texture", "count", "shape")},
        pair_both_correct=0.98,
    )
    control = {
        n: dict(overall=dict(accuracy=0.5, nll=1.0, brier=0.5, ece=0.02)) for n in CONDITIONS
    }
    candidate = dict.fromkeys(CONDITIONS, good)
    intervals = {n: dict(lower=[0.4, -1.0]) for n in CONDITIONS}
    assert quality_failures(candidate, control, intervals) == []
    candidate["complement_descriptions"] = dict(
        good, tasks=dict(good["tasks"], color=dict(accuracy=0.94))
    )
    assert "complement_descriptions:minimum_task_accuracy" in quality_failures(
        candidate, control, intervals
    )
