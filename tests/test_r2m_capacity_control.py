import pytest

from scripts.compare_r2m_capacity_control import validate_run_pair


def _run_pair():
    common = {"choices": ["a", "b"], "learning_rate": 0.001, "seed": 17}
    fusion_config = {
        **common,
        "model_arms": ["audio_only", "audio_sensor_gated"],
        "modality_condition": "normal",
        "source_commit": "old",
        "source_provenance": {"hash": "old"},
        "effective_config_sha256_excluding_self": "old",
    }
    control_config = {
        **common,
        "model_arms": ["audio_sensor_gated"],
        "modality_condition": "without_sensor",
        "source_commit": "new",
        "source_provenance": {"hash": "new"},
        "effective_config_sha256_excluding_self": "new",
    }
    fusion_results = {"arms": [{
        "mode": "audio_sensor_gated",
        "parameter_count": 180387,
        "modality_condition": "normal",
    }]}
    control_results = {"arms": [{
        "mode": "audio_sensor_gated",
        "parameter_count": 180387,
        "modality_condition": "without_sensor",
    }]}
    return fusion_results, fusion_config, control_results, control_config


def test_capacity_control_requires_equal_config_and_total_parameters():
    pair = _run_pair()
    fusion_arm, control_arm = validate_run_pair(*pair)

    assert fusion_arm["parameter_count"] == control_arm["parameter_count"] == 180387


def test_capacity_control_rejects_unplanned_training_change():
    fusion_results, fusion_config, control_results, control_config = _run_pair()
    control_config["learning_rate"] = 0.002

    with pytest.raises(ValueError, match="differ outside the declared condition"):
        validate_run_pair(
            fusion_results, fusion_config, control_results, control_config
        )


def test_capacity_control_rejects_parameter_count_mismatch():
    fusion_results, fusion_config, control_results, control_config = _run_pair()
    control_results["arms"][0]["parameter_count"] = 999

    with pytest.raises(ValueError, match="parameter counts differ"):
        validate_run_pair(
            fusion_results, fusion_config, control_results, control_config
        )
