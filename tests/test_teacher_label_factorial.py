import pytest

from scripts.build_conditional_curriculum import build
from scripts.condition_binding_corpus import all_states, questions_for
from scripts.semantic_teacher_probe import LABELS, build_probe, input_text
from scripts.teacher_label_factorial import (
    build_factorial,
    factorial_summary,
    render_input,
    summarize_factorial_rows,
)


def sources():
    return [r for s in all_states()[::19][:8] for r in questions_for(s, "train")], build(
        17, 8, "train"
    )


def test_complete_2_by_2_preserves_meaning_and_old_endpoints():
    binding, numeric = sources()
    old = {r["id"]: r for r in build_probe(binding, numeric, 4)}
    rows = build_factorial(binding, numeric, 4)
    assert len(rows) == len({r["id"] for r in rows}) == 192
    assert len({(r["source"], r["group"]) for r in rows}) == 8
    for start in range(0, len(rows), 4):
        block = rows[start : start + 4]
        assert [r["arm"] for r in block] == ["00", "10", "01", "11"]
        original = block[0]
        canonical = original["option_ids"]
        for row in block:
            assert row["state"] == original["state"]
            assert row["question"] == original["question"]
            assert row["target_id"] == original["target_id"]
            assert dict(zip(row["option_ids"], row["options"], strict=True)) == dict(
                zip(original["option_ids"], original["options"], strict=True)
            )
            order, mapping = map(int, row["arm"])
            assert row["option_ids"] == canonical[:: -1 if order else 1]
            expected = dict(
                zip(canonical, list(LABELS[: len(canonical)])[:: -1 if mapping else 1], strict=True)
            )
            assert dict(zip(row["option_ids"], row["labels"], strict=True)) == expected
            if row["baseline_row_id"]:
                assert render_input(row) == input_text(old[row["baseline_row_id"]])


def test_nontrain_and_duplicate_sources_remain_rejected():
    binding, numeric = sources()
    with pytest.raises(ValueError):
        build_factorial([dict(binding[0], split="validation")] + binding[1:], numeric, 4)
    with pytest.raises(ValueError):
        build_factorial(binding + binding[:1], numeric, 4)


def test_render_excludes_target_identity_and_rejects_ambiguous_labels():
    row = build_factorial(*sources(), 4)[0]
    changed = dict(row, target="SECRET", target_id="SECRET", group="SECRET", task="SECRET")
    assert render_input(row) == render_input(changed)
    with pytest.raises(ValueError):
        render_input(dict(row, labels=["A"] * len(row["labels"])))
    with pytest.raises(ValueError):
        render_input(dict(row, labels=row["labels"][:-1]))


def test_zero_factor_effects_for_semantically_identical_predictions():
    rows = build_factorial(*sources(), 4)
    predictions = []
    for row in rows:
        predictions.append(
            dict(
                row_id=row["id"],
                logits=[20.0 if oid == row["target_id"] else 0.0 for oid in row["option_ids"]],
            )
        )
    result = factorial_summary(rows, predictions)
    metrics = summarize_factorial_rows(rows, predictions)
    assert len(metrics["conditions"]) == 8
    assert all(v["overall"]["accuracy"] == 1 for v in metrics["conditions"].values())
    assert all(v["answer_agreement"] == 1 for v in metrics["reversal"].values())
    for source in ("binding", "numeric"):
        assert result[source]["groups"] == 4
        for contrast in result[source]["contrasts"].values():
            assert contrast["point"] == pytest.approx([0.0, 0.0, 0.0], abs=1e-12)
        for pair in result[source]["invariance"].values():
            assert pair["answer_agreement"] == 1.0
            assert pair["maximum_probability_difference"] < 1e-12
    with pytest.raises(ValueError):
        factorial_summary(rows, predictions[:-1])


def test_summary_identifies_only_display_order_effect_in_controlled_scores():
    rows = build_factorial(*sources(), 4)
    predictions = []
    for row in rows:
        winner = row["target_id"]
        if row["arm"][0] == "1":
            winner = next(oid for oid in row["option_ids"] if oid != winner)
        predictions.append(
            dict(
                row_id=row["id"],
                logits=[20.0 if oid == winner else 0.0 for oid in row["option_ids"]],
            )
        )
    result = factorial_summary(rows, predictions)
    for source in ("binding", "numeric"):
        c = result[source]["contrasts"]
        assert c["order_with_original_mapping"]["point"][0] == -1
        assert c["order_with_reversed_mapping"]["point"][0] == -1
        assert c["mapping_with_original_order"]["point"][0] == 0
        assert c["mapping_with_reversed_order"]["point"][0] == 0
        assert c["interaction"]["point"][0] == 0
