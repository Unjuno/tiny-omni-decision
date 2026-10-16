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
