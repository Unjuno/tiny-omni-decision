from __future__ import annotations

from pathlib import Path

import yaml

from tiny_omni_decision.haptile_action import (
    ACTION_LABELS,
    ACTION_OPTION_TEXTS,
)
from tiny_omni_decision.haptile_decision import HAPTILE_MODES

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_haptile_decision_config_freezes_action_and_quality_protocol():
    config = yaml.safe_load(
        (REPO_ROOT / "configs/pretrained_reuse/haptile_temporal_decision.yaml")
        .read_text(encoding="utf-8")
    )

    assert config["seed"] == 17
    assert config["device"] == "cpu"
    assert config["data"]["sealed_audit_loading"] is False
    assert config["data"]["video_num_frames"] == 8
    assert config["data"]["target_deadband_m"] == 0.005
    assert [row["id"] for row in config["data"]["options"]] == list(ACTION_LABELS)
    assert [row["text"] for row in config["data"]["options"]] == list(
        ACTION_OPTION_TEXTS
    )
    assert config["readout"]["trainable_parameters"] == 519_683
    assert config["readout"]["total_dense_parameters"] == 132_274_691
    assert config["readout"]["text_only_control_parameters"] == 196_865
    assert config["readout"]["all_comparison_heads_parameters"] == 716_548
    assert config["readout"]["modes"] == list(HAPTILE_MODES)
    assert config["training"]["checkpoint_selection"] == (
        "minimum_validation_macro_task_group_nll"
    )
    assert config["prohibitions"]["sealed_audit_loading"] is False
