import copy

import pytest

from scripts.build_conditional_curriculum import build
from scripts.condition_binding_corpus import all_states, questions_for
from scripts.semantic_teacher_probe import build_probe, input_text, summarize_rows


def sources():
    return [r for s in all_states()[::19][:8] for r in questions_for(s, "train")], build(
        17, 8, "train"
    )


def test_probe_is_train_only_deterministic_complete_and_preserves_options():
    binding, numeric = sources()
    rows = build_probe(binding, numeric, 4)
    assert rows == build_probe(binding[::-1], numeric[::-1], 4)
    assert len(rows) == len({r["id"] for r in rows}) == 144
    assert {r["split"] for r in rows} == {"train"}
    assert len({(r["source"], r["group"]) for r in rows}) == 8
    assert {len(r["options"]) for r in rows if r["source"] == "binding"} == {40}
    for row in rows:
        assert row["options"][row["option_ids"].index(row["target_id"])] == row["target"]
    for row in rows:
        if row["condition"] == "exclusion":
            reverse = next(
                r
                for r in rows
                if r["id"] == row["id"].replace(":exclusion:", ":reverse_exclusion:")
            )
            assert row["option_ids"] == reverse["option_ids"][::-1]
            assert row["question"] == reverse["question"]


def test_nontrain_incomplete_duplicate_or_wrong_oracle_fails():
    binding, numeric = sources()
    for bad in (
        [dict(binding[0], split="validation")] + binding[1:],
        binding[:-1],
        binding + binding[:1],
    ):
        with pytest.raises(ValueError):
            build_probe(bad, numeric, 4)
    broken = copy.deepcopy(numeric)
    broken[0]["target"] = "invalid"
    with pytest.raises(ValueError):
        build_probe(binding, broken, 4)


def test_prompt_exposes_only_inputs_not_oracle_or_identity():
    binding, numeric = sources()
    row = build_probe(binding, numeric, 4)[0]
    text = input_text(row)
    altered = dict(
        row,
        target="SENTINEL",
        target_id="SENTINEL",
        group="SENTINEL",
        task="SENTINEL",
        oracle="SENTINEL",
    )
    assert input_text(altered) == text
    assert "SENTINEL" not in text
    assert all(f"{label}. " in text for label in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmn")


def test_uniform_and_perfect_metric_and_reversal_accounting():
    binding, numeric = sources()
    rows = build_probe(binding, numeric, 4)
    outputs = []
    for row in rows:
        n = len(row["options"])
        target = row["option_ids"].index(row["target_id"])
        outputs.append(
            dict(row_id=row["id"], logits=[20.0 if i == target else 0.0 for i in range(n)])
        )
    report = summarize_rows(rows, outputs)
    assert report["conditions"]["binding/direct"]["overall"]["accuracy"] == 1
    assert report["conditions"]["binding/exclusion"]["pair_both_correct"] == 1
    assert report["reversal"]["binding"]["answer_agreement"] == 1
    assert report["reversal"]["numeric"]["answer_agreement"] == 1
    with pytest.raises(ValueError):
        summarize_rows(rows, outputs[:-1])
