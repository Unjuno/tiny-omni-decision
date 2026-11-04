"""The spatial intervention cannot silently change the learning experiment."""

import json
from pathlib import Path


def test_full_extent_config_keeps_original_training_and_data_conditions():
    repo = Path(__file__).resolve().parents[1]
    original = json.loads(
        (repo / "configs/pretrained_reuse/native_long_budget_v1.json").read_text()
    )
    candidate = json.loads(
        (repo / "configs/pretrained_reuse/native_full_extent_fit_v1.json").read_text()
    )
    for key in (
        "training",
        "corpus",
        "queries",
        "initial_readout",
        "parent_sampling",
        "pools",
        "plans",
        "state_plan",
        "union_probe",
        "training_probe_scope",
    ):
        assert candidate[key] == original[key], key
    assert candidate["intervention"] == "full-extent-bilinear-centered-black-pad-288-v1"
    assert candidate["control_fit"].replace("\\", "/").endswith("/native-long-budget-fit-01")
    assert candidate["visual_cache"].replace("\\", "/").endswith("/native-full-extent-cache-01")
    assert original["input_pins"].items() <= candidate["input_pins"].items()
    assert len(candidate["input_pins"]) == len(original["input_pins"]) + 2


def test_fresh_full_extent_long_config_is_separate_and_pins_same_schedule():
    repo = Path(__file__).resolve().parents[1]
    repeat = json.loads(
        (repo / "configs/pretrained_reuse/native_full_extent_fit_v1.json").read_text()
    )
    fresh = json.loads(
        (
            repo
            / "configs/pretrained_reuse/native_full_extent_unique_coverage_long_v1.json"
        ).read_text()
    )
    assert fresh["id"] == "native-full-extent-unique-coverage-long-v1"
    assert fresh["training"] == repeat["training"]
    for key in ("corpus", "queries", "initial_readout", "parent_sampling", "state_plan"):
        assert fresh[key] == repeat[key]
    assert fresh["visual_cache"] == repeat["visual_cache"]
    assert fresh["intervention"] == repeat["intervention"]
    assert fresh["pools"]["repeat"] == repeat["pools"]["repeat"]
    assert fresh["plans"]["repeat"] == repeat["plans"]["repeat"]
    assert fresh["pools"]["fresh"].replace("\\", "/").endswith(
        "/native-coverage-plan-02/fresh-pool.json"
    )
    assert fresh["plans"]["fresh"].replace("\\", "/").endswith(
        "/native-coverage-plan-02/fresh-plan.json"
    )
    assert fresh["fresh_control_fit"].replace("\\", "/").endswith(
        "/native-coverage-long-fresh-fit-01"
    )
    assert fresh["common_train_probe_scope"] == "parent-6656"
    assert fresh["input_pins"].items() >= repeat["input_pins"].items()


def test_parent_probe_is_same_corpus_order_for_duplicate_occurrences():
    from scripts.native_full_extent import parent_probe

    rows = [
        {"id": "row-0", "split": "train"},
        {"id": "row-1", "split": "train"},
        {"id": "row-2", "split": "train"},
    ]
    assert parent_probe(rows, [2, 0, 2, 1]) == rows
    assert parent_probe(rows, [2, 0, 2]) == [rows[0], rows[2]]
