# ruff: noqa: E402

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from scripts.evaluate_physionpp_scenario_holdout_path_a import (
    leave_one_scenario_out,
    scenario_name,
)


def _row(index: int, scenario: str, seed: int) -> dict[str, str]:
    return {
        "id": f"sample-{index}",
        "source_asset_group_id": f"{scenario}:seed={seed}",
    }


def test_scenario_split_holds_out_every_row_and_scene_group_once() -> None:
    rows = [
        _row(0, "scenario-a", 1),
        _row(1, "scenario-a", 2),
        _row(2, "scenario-b", 1),
        _row(3, "scenario-b", 2),
        _row(4, "scenario-c", 1),
    ]
    folds = leave_one_scenario_out(rows)
    assert set(folds) == {"scenario-a", "scenario-b", "scenario-c"}
    held_out = []
    for scenario, (train, validation) in folds.items():
        assert all(scenario_name(rows[index]) == scenario for index in validation)
        assert all(scenario_name(rows[index]) != scenario for index in train)
        assert {rows[index]["source_asset_group_id"] for index in train}.isdisjoint(
            rows[index]["source_asset_group_id"] for index in validation
        )
        held_out.extend(validation)
    assert sorted(held_out) == list(range(len(rows)))


def test_scenario_parser_rejects_unverified_group_identity() -> None:
    with pytest.raises(ValueError, match="encode scenario:seed"):
        scenario_name({"source_asset_group_id": "unknown-group-format"})


def test_scenario_split_requires_multiple_scenarios() -> None:
    with pytest.raises(ValueError, match="at least two"):
        leave_one_scenario_out([_row(0, "scenario-a", 1)])
