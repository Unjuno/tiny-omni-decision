import pytest

from scripts.native_text_state_holdout import (
    SOURCE,
    balanced_order,
    build_split,
    counterfactual_donor_indices,
)
from scripts.train_native_text_group_holdout import (
    configured_source_split,
    validate_config,
    validate_split_membership,
)


def make_rows(group_count=10):
    rows = []
    for group in range(group_count):
        for task in ("color", "shape"):
            target = (group + (task == "shape")) % 3
            options = ["red", "blue", "green"]
            rows.append(
                {
                    "id": f"{group}:{task}",
                    "split": "train",
                    "source": SOURCE,
                    "task": task,
                    "group": f"group-{group}",
                    "observation": f"text:group-{group}",
                    "question": f"Which {task}?",
                    "options": options,
                    "option_ids": options,
                    "queries": [f"query-{i}" for i in range(3)],
                    "target_index": target,
                }
            )
    return rows


def test_split_is_deterministic_grouped_and_excludes_previous_sampler_groups():
    rows = make_rows()
    consumed = {"0:color", "1:shape", "2:color", "3:shape"}
    first = build_split(rows, consumed, holdout_fraction=0.5)
    second = build_split(rows, consumed, holdout_fraction=0.5)

    assert first["holdout_groups"] == second["holdout_groups"]
    assert {r["group"] for r in first["fit_rows"]}.isdisjoint(first["holdout_groups"])
    assert {r["group"] for r in first["holdout_rows"]} == set(first["holdout_groups"])
    assert not {r["observation"] for r in first["fit_rows"]} & {
        r["observation"] for r in first["holdout_rows"]
    }
    assert set(first["holdout_groups"]).isdisjoint(first["previously_seen_groups"])
    assert first["holdout_task_counts"] == {
        "color": len(first["holdout_groups"]),
        "shape": len(first["holdout_groups"]),
    }


def test_split_rejects_validation_rows_and_duplicate_ids():
    rows = make_rows()
    rows[0]["split"] = "validation"
    with pytest.raises(ValueError, match="TRAIN"):
        build_split(rows, set())

    rows = make_rows()
    rows[1]["id"] = rows[0]["id"]
    with pytest.raises(ValueError, match="unique"):
        build_split(rows, set())


def test_balanced_order_is_fixed_four_per_update_and_deterministic():
    rows = make_rows()
    first = balanced_order(rows, 9)
    assert first == balanced_order(rows, 9)
    assert len(first) == 9
    assert all(len(batch) == 4 for batch in first)
    assert all(0 <= index < len(rows) for batch in first for index in batch)


def test_counterfactual_donors_are_same_task_different_group_and_label():
    rows = make_rows()
    donors = counterfactual_donor_indices(rows)
    assert len(donors) == len(rows)
    for index, donor_index in donors.items():
        row, donor = rows[index], rows[donor_index]
        assert row["task"] == donor["task"]
        assert row["question"] == donor["question"]
        assert row["group"] != donor["group"]
        assert set(row["option_ids"]) == set(donor["option_ids"])
        assert row["option_ids"][row["target_index"]] != donor["option_ids"][donor["target_index"]]


def test_split_fails_closed_when_a_task_has_only_seen_groups():
    rows = make_rows()
    rows = [r for r in rows if r["task"] == "color"]
    with pytest.raises(ValueError, match="untouched"):
        build_split(rows, {r["id"] for r in rows})


def test_split_supports_an_explicit_distinct_train_source_and_split_identity():
    source = "text:programmatic_decision_v1"
    split_id = "native-text-programmatic-group-holdout-v1"
    rows = make_rows(group_count=20)
    programmatic = []
    for row in rows:
        changed = dict(row)
        changed["source"] = source
        programmatic.append(changed)

    first = build_split(
        programmatic,
        {"0:color", "1:shape"},
        seed=17,
        holdout_fraction=0.5,
        source=source,
        split_id=split_id,
    )
    second = build_split(
        programmatic,
        {"0:color", "1:shape"},
        seed=17,
        holdout_fraction=0.5,
        source=source,
        split_id=split_id,
    )

    assert first["source"] == source
    assert first["split_id"] == split_id
    assert first["holdout_groups"] == second["holdout_groups"]
    assert {row["source"] for row in first["fit_rows"] + first["holdout_rows"]} == {source}
    assert set(first["fit_groups"]).isdisjoint(first["holdout_groups"])
    assert len(first["holdout_groups"]) == 9
    assert balanced_order(first["fit_rows"], 7, split_id=split_id) == balanced_order(
        first["fit_rows"], 7, split_id=split_id
    )


def test_split_identity_changes_deterministic_group_ranking():
    rows = make_rows(group_count=20)
    consumed = {"0:color", "1:shape"}
    first = build_split(rows, consumed, split_id="split-a")
    second = build_split(rows, consumed, split_id="split-b")
    assert first["holdout_groups"] != second["holdout_groups"]


def test_runner_accepts_legacy_and_programmatic_configs_without_mixing_profiles():
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    legacy = json.loads(
        (root / "configs/pretrained_reuse/native_text_state_group_holdout_v1.json").read_text(
            encoding="utf-8"
        )
    )
    programmatic = json.loads(
        (
            root / "configs/pretrained_reuse/native_text_programmatic_group_holdout_v1.json"
        ).read_text(encoding="utf-8")
    )

    assert configured_source_split(legacy) == (SOURCE, "native-text-state-group-holdout-v1")
    assert configured_source_split(programmatic) == (
        "text:programmatic_decision_v1",
        "native-text-programmatic-group-holdout-v1",
    )
    validate_config(legacy)
    validate_config(programmatic)

    programmatic["split"]["heldout_group_count"] = 65
    with pytest.raises(ValueError, match="preregistered"):
        validate_config(programmatic)


def test_training_rechecks_group_and_observation_split_integrity():
    rows = make_rows(group_count=4)
    fit = [row for row in rows if row["group"] in {"group-0", "group-1"}]
    holdout = [row for row in rows if row["group"] in {"group-2", "group-3"}]
    validate_split_membership(fit, holdout, {"group-0", "group-1"}, SOURCE)

    leaked_group = dict(holdout[0], group="group-0")
    with pytest.raises(ValueError, match="groups overlap"):
        validate_split_membership(fit, [leaked_group, *holdout[1:]], set(), SOURCE)

    seen_group = dict(holdout[0])
    with pytest.raises(ValueError, match="previously consumed"):
        validate_split_membership(fit, holdout, {seen_group["group"]}, SOURCE)
