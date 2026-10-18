import json
import math
from pathlib import Path

import pytest

from scripts.train_greatest_hits_av_match_readouts import fit_temperature, probability_metrics


def test_preregistered_readout_config_is_valid_json_and_cpu_only():
    config_path = Path("configs/pretrained_reuse/greatest_hits_av_match_readouts_v0.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))

    assert config["optimizer"]["device"] == "cpu"
    assert config["optimizer"]["epochs"] == 100
    assert config["evaluation"]["sealed_audit_opened"] is False


def test_binary_decision_metrics_use_probability_options_in_fixed_order():
    metrics = probability_metrics([0, 1], [[0.9, 0.1], [0.1, 0.9]])

    assert metrics["count"] == 2
    assert metrics["accuracy"] == 1.0
    assert metrics["nll"] == pytest.approx(-math.log(0.9))
    assert metrics["brier"] == pytest.approx(0.02)
    assert metrics["ece_15"] == pytest.approx(0.1)
    assert metrics["mean_confidence"] == pytest.approx(0.9)


def test_uniform_probabilities_are_calibrated_and_bad_vectors_fail_closed():
    metrics = probability_metrics([0, 1], [[0.5, 0.5], [0.5, 0.5]])

    assert metrics["accuracy"] == 0.5
    assert metrics["nll"] == pytest.approx(math.log(2))
    assert metrics["brier"] == pytest.approx(0.5)
    assert metrics["ece_15"] == 0.0
    with pytest.raises(ValueError, match="sum to one"):
        probability_metrics([0], [[0.4, 0.4]])


def test_temperature_fit_accepts_logits_created_in_inference_mode():
    torch = pytest.importorskip("torch", reason="PyTorch is an optional ML dependency")

    with torch.inference_mode():
        logits = torch.tensor([[3.0, -3.0], [-1.0, 1.0], [0.5, -0.5]])
    temperature = fit_temperature(logits, torch.tensor([0, 1, 0]))

    assert math.isfinite(temperature)
    assert temperature > 0
