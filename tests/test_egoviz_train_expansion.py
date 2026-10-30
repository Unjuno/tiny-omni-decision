import hashlib
import re
from pathlib import Path

import pytest

from scripts.prepare_egoviz_train_expansion import select_expansion


def test_expansion_is_deterministic_balanced_and_excludes_all_reserved_groups():
    records = [
        {
            "group_id": f"g{i}",
            "chunk_id": f"c{i}",
            "category": f"category{i % 2}",
            "duration_s": 16.0,
            "mcap_bytes": 100 + i,
            "mcap_path": f"data/c{i}.mcap",
        }
        for i in range(30)
    ]
    a = select_expansion(records, {"g0", "g1"}, count=24, seed=17)
    assert a == select_expansion(records, {"g0", "g1"}, count=24, seed=17)
    assert len({r["group_id"] for r in a}) == 24
    assert not ({r["group_id"] for r in a} & {"g0", "g1"})
    assert all(
        sum(r["category"] == category for r in a) >= 2 for category in {"category0", "category1"}
    )


def test_expansion_fails_closed_when_distinct_groups_are_insufficient():
    row = {
        "group_id": "g1",
        "chunk_id": "c1",
        "category": "one",
        "duration_s": 16.0,
        "mcap_bytes": 100,
        "mcap_path": "data/c1.mcap",
    }
    with pytest.raises(ValueError, match="distinct"):
        select_expansion([row], set(), count=2, seed=17)


def test_controller_pins_the_actual_frozen_plan_hash():
    repo = Path(__file__).resolve().parents[1]
    plan = repo / "manifests/pretrained_reuse/egoviz-train-expansion-plan-20261030.json"
    controller = repo / "scripts/run_egoviz_train_expansion.ps1"
    match = re.search(r"\.Hash -ine '([0-9a-f]+)'", controller.read_text(encoding="utf-8"))
    assert match is not None
    assert match.group(1) == hashlib.sha256(plan.read_bytes()).hexdigest()
