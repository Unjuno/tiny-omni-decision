from __future__ import annotations

import json
from pathlib import Path

from tiny_omni_decision.decision_json import decision_json_record

REPO_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = REPO_ROOT / "configs" / "pretrained_reuse" / "decision_task_contract_v0_1.json"


def test_task_contract_keeps_product_modalities_and_battery_state_explicit() -> None:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))

    assert contract["status"] == "engineering-interface-frozen_task_battery-open"
    assert contract["product_inputs"]["required_modalities"] == [
        "text",
        "image",
        "audio",
        "video",
    ]
    assert contract["product_inputs"]["optional_modalities"] == ["sensor"]
    assert contract["evaluation_policy"]["task_battery_status"] == "open"
    assert contract["evaluation_policy"]["minimum_examples_per_named_task"] == 1000
    assert contract["evaluation_policy"]["minimum_independent_groups_per_named_task"] == 200
    assert {task["modality"] for task in contract["known_development_tasks"]} == {
        "text",
        "image",
        "audio",
        "video",
    }
    assert all(
        task["status"].startswith("development_only")
        for task in contract["known_development_tasks"]
    )


def test_task_contract_decision_types_match_executable_json_schema() -> None:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    decision_types = contract["request_contract"]["decision_type"]
    cases = {
        "choice": (["a", "b"], ["A", "B"], [0.75, 0.25], None),
        "yes_no": (["yes", "no"], ["Yes", "No"], [0.75, 0.25], None),
        "ordinal_score": (
            ["low", "mid", "high"],
            ["Low", "Medium", "High"],
            [0.2, 0.3, 0.5],
            [1, 2, 3],
        ),
    }

    assert set(decision_types) == set(cases)
    for decision_type in decision_types:
        option_ids, option_texts, probabilities, ordinal_values = cases[decision_type]
        record = decision_json_record(
            request_id="contract-test",
            decision_type=decision_type,
            option_ids=option_ids,
            option_texts=option_texts,
            probabilities=probabilities,
            ordinal_values=ordinal_values,
        )
        assert record["schema_version"] == contract["response_contract"]["schema_version"]
        assert [option["id"] for option in record["options"]] == option_ids
        assert [option["probability"] for option in record["options"]] == probabilities
        assert record["selected_option_id"] == option_ids[probabilities.index(max(probabilities))]
