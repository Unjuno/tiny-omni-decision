from pathlib import Path

from tiny_omni_decision.io import load_structured_file

ROOT = Path(__file__).resolve().parents[1]


def test_text_decision_lora_config_parses_and_keeps_architecture_targets_unset() -> None:
    config = load_structured_file(ROOT / "configs" / "decision" / "e2b_qat_lora.yaml")

    assert config["schema_version"] == 1
    assert config["task"]["type"] == "typed_decision"
    assert config["training"]["method"] == "lora"
    assert config["training"]["rank"] == 16
    assert config["training"]["alpha"] == 32
    assert config["training"]["target_modules"] == []
    assert config["loss"]["cross_entropy"] == 1.0
    assert config["loss"]["brier"] == 0.2
