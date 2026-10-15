"""Behavioral tests: independent labels, leakage gates and selection safeguards."""

import copy
import hashlib
import json

import pytest

from scripts import decision_supervision as subject
from scripts.condition_binding_corpus import all_states


def test_binding_separates_true_irrelevant_from_relevant_false():
    state = all_states()[0]
    rows = subject.binding_rows([state], "train")
    assert len(rows) == 16
    row = rows[0]
    assert len(row["options"]) == 40
    labels = row["supervision"]
    assert sum(labels["fact_true"]) == 4
    assert sum(labels["request_relevant"]) == 10
    assert sum(a and b for a, b in zip(
        labels["fact_true"], labels["request_relevant"], strict=True)) == 1
    assert sum(a and not b for a, b in zip(
        labels["fact_true"], labels["request_relevant"], strict=True)) == 3
    assert row["target_id"] == f"color:{state['color']}"
    for row in rows:
        subject.validate_row(row)


def numeric_fixture():
    return [
        dict(id="K", cost=9, risk=3, time=5),
        dict(id="M", cost=2, risk=7, time=8),
        dict(id="P", cost=6, risk=1, time=2),
        dict(id="R", cost=4, risk=9, time=4),
    ]


def test_numeric_targets_are_independently_checkable_with_all_choices():
    rows = subject.numeric_rows([numeric_fixture()], "train")
    assert len(rows) == 8
    assert [r["target_id"] for r in rows[:6]] == ["M", "K", "P", "R", "P", "M"]
    assert [len(r["options"]) for r in rows] == [4] * 6 + [5, 5]
    for row in rows:
        subject.validate_row(row)
        assert row["supervision"]["fact_true"] is None


def test_numeric_identity_ignores_order_names_and_positive_affine_change():
    original = numeric_fixture()
    changed = [dict(id=str(i), **{k: 3 * item[k] + 11 for k in ("cost", "risk", "time")})
               for i, item in enumerate(reversed(original))]
    assert subject.state_key(original) == subject.state_key(changed)
    changed[0]["cost"] += 1
    assert subject.state_key(original) != subject.state_key(changed)


def test_numeric_aliases_do_not_bypass_historical_overlap():
    original = numeric_fixture()
    aliases = [dict(candidate=str(i), price=s["cost"], risk=s["risk"], time=s["time"])
               for i, s in enumerate(original)]
    assert subject.state_key(original) == subject.state_key(aliases)


def test_plain_text_states_use_normalized_content_identity():
    assert subject.state_key("  Cost is low.\nRisk is high. ") == subject.state_key(
        "Cost is low. Risk is high.")
    assert subject.state_key("Cost is low.") != subject.state_key("Cost is high.")


def test_numeric_boundary_none_and_unknown_rules():
    state = numeric_fixture()
    assert subject.numeric_solution(state, dict(operation="count", threshold=3)) == "2"
    assert subject.numeric_solution(state, dict(operation="filter_min_cost", threshold=3)) == "P"
    assert subject.numeric_solution(state, dict(operation="filter_min_cost", threshold=0)) == "none"
    for rule in (dict(operation="guess", threshold=3),
                 dict(operation="extremum", field="risk", direction="middle")):
        with pytest.raises(ValueError):
            subject.numeric_solution(state, rule)


@pytest.mark.parametrize("damage", ["target", "question", "labels", "candidate"])
def test_semantic_corruption_and_candidate_deletion_fail_closed(damage):
    row = subject.binding_rows([all_states()[0]], "train")[0]
    if damage == "target":
        row["target_id"] = "wrong"
    elif damage == "question":
        row["question"] = "Answer an unrelated question."
    elif damage == "labels":
        row["supervision"]["fact_true"] = [False] * 40
    else:
        row["options"].pop()
        row["option_ids"].pop()
    with pytest.raises(ValueError):
        subject.validate_row(row)


def test_public_input_cannot_read_gold_or_task_routing():
    row = subject.binding_rows([all_states()[0]], "train")[0]
    expected = {k: copy.deepcopy(row[k]) for k in ("state", "question", "options", "option_ids")}
    for key in ("supervision", "target", "target_id", "task", "rule", "group"):
        row[key] = "POISONED_PRIVATE_VALUE"
    assert subject.public_example(row) == expected
    subject.public_example(row)["options"].clear()
    assert len(row["options"]) == 40


def test_partition_gate_catches_state_and_exclusion_overlap():
    state = all_states()[0]
    train = subject.binding_rows([state], "train")
    validation = subject.binding_rows([state], "validation")
    with pytest.raises(ValueError, match="state overlap"):
        subject.verify_partitions(dict(train=train, validation=validation))
    with pytest.raises(ValueError, match="observed state"):
        subject.verify_partitions(dict(train=train), [subject.state_key(state)])


def test_template_splits_and_generation_are_deterministic():
    states = all_states()[:3]
    parts = {s: subject.binding_rows([state], s)
             for s, state in zip(("train", "validation", "transfer"), states, strict=True)}
    assert parts["train"] == subject.binding_rows([states[0]], "train")
    assert [len(p) for p in parts.values()] == [16, 8, 8]
    assert subject.verify_partitions(parts)["state_overlap"] == 0
    assert not ({r["question"] for r in parts["train"]}
                & {r["question"] for r in parts["validation"]})


def metrics(accuracy=.8, nll=.5, brier=.3, ece=.04):
    return dict(accuracy=accuracy, nll=nll, brier=brier, ece=ece, count=10)


def schema(baseline):
    return {t: m["count"] for t, m in baseline.items()}


def test_selector_rejects_missing_tasks_damage_nonfinite_and_transfer():
    baseline = dict(binding=metrics(), numeric=metrics(), original=metrics())
    good = dict(binding=metrics(.85, .4), numeric=metrics(.82, .45), original=metrics())
    history = [dict(step=0, purpose="validation", metrics=baseline),
               dict(step=128, purpose="validation", metrics=good),
               dict(step=256, purpose="validation", metrics=good)]
    assert subject.select_validation(history, baseline, schema(baseline)) == 128
    bad = copy.deepcopy(good)
    bad["original"]["accuracy"] = .78
    history[1]["metrics"] = bad
    assert subject.select_validation(history, baseline, schema(baseline)) == 256
    bad["original"]["accuracy"] = float("nan")
    history[2]["metrics"] = bad
    assert subject.select_validation(history, baseline, schema(baseline)) == 0
    for damage in ("missing", "transfer"):
        h = copy.deepcopy(history)
        if damage == "missing":
            del h[1]["metrics"]["numeric"]
        else:
            h[1]["purpose"] = "transfer"
        with pytest.raises(ValueError):
            subject.select_validation(h, baseline, schema(baseline))


def test_fit_loader_rejects_transfer_before_opening_any_file(tmp_path):
    with pytest.raises(ValueError, match="partition"):
        subject.load_for_fit(tmp_path / "missing", "transfer", "0" * 64)


def test_fit_loader_checks_manifest_payload_and_partition(tmp_path):
    rows = subject.binding_rows([all_states()[0]], "train")
    path = tmp_path / "train.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    manifest = tmp_path / "result.json"
    manifest.write_text(json.dumps(dict(status="PASS_FROZEN_DECISION_SUPERVISION",
        artifacts={path.name: hashlib.sha256(path.read_bytes()).hexdigest()})))
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert subject.load_for_fit(tmp_path, "train", digest) == rows
    with pytest.raises(ValueError, match="manifest hash"):
        subject.load_for_fit(tmp_path, "train", "0" * 64)
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="partition hash"):
        subject.load_for_fit(tmp_path, "train", digest)


def test_selector_keeps_parent_when_eligible_fit_has_worse_macro_nll():
    baseline = dict(binding=metrics(), numeric=metrics())
    worse = dict(binding=metrics(.9, .51), numeric=metrics(.9, .51))
    history = [dict(step=0, purpose="validation", metrics=baseline),
               dict(step=128, purpose="validation", metrics=worse)]
    assert subject.select_validation(history, baseline, schema(baseline)) == 0
    history[1]["metrics"]["numeric"]["count"] = 9
    with pytest.raises(ValueError, match="counts"):
        subject.select_validation(history, baseline, schema(baseline))


def test_selector_binds_baseline_counts_to_manifest_and_metric_domains():
    baseline = dict(binding=metrics())
    history = [dict(step=0, purpose="validation", metrics=baseline)]
    with pytest.raises(ValueError):
        subject.select_validation(history, baseline, {"binding": 128})
    for key, value in (("accuracy", 2), ("nll", -1), ("brier", -1), ("ece", -1)):
        bad = copy.deepcopy(baseline)
        bad["binding"][key] = value
        with pytest.raises(ValueError):
            subject.select_validation(history + [dict(step=128, purpose="validation",
                                                       metrics=bad)], baseline, schema(baseline))


def test_empty_and_unknown_partition_rejected():
    with pytest.raises(ValueError):
        subject.binding_rows([], "train")
    with pytest.raises(ValueError):
        subject.numeric_rows([numeric_fixture()], "audit")
    with pytest.raises(ValueError):
        subject.verify_partitions({})


def test_public_numeric_records_remain_actual_numbers():
    row = subject.numeric_rows([numeric_fixture()], "train")[0]
    assert json.loads(subject.public_example(row)["state"])[0]["cost"] == 9
