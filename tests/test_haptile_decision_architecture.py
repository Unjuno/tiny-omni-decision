from __future__ import annotations

import torch

from tiny_omni_decision.video_sensor_temporal_decision import (
    VideoSensorTemporalDecisionScorer,
)


def test_haptile_video_sensor_decision_shapes_and_parameter_budget():
    model = VideoSensorTemporalDecisionScorer(
        video_feature_width=3072,
        sensor_channels=25,
        option_width=1536,
        context_width=384,
        sensor_hidden_size=32,
        width=64,
        heads=4,
        scorer_hidden_width=64,
    )
    output = model(
        torch.randn(2, 8, 3072),
        torch.arange(8, dtype=torch.float32).repeat(2, 1) * 1000 / 7,
        torch.ones(2, 8, dtype=torch.bool),
        torch.ones(2, dtype=torch.bool),
        torch.randn(2, 8, 25),
        torch.arange(8, dtype=torch.float32).repeat(2, 1) * 1000 / 7,
        torch.ones(2, 8, dtype=torch.bool),
        torch.ones(2, dtype=torch.bool),
        torch.randn(2, 7, 1536),
        torch.ones(2, 7, dtype=torch.bool),
        context_features=torch.randn(2, 384),
    )

    assert output.shape == (2, 7)
    assert torch.isfinite(output).all()
    assert model.parameter_count == 519_683
    assert 114_365_184 + 17_389_824 + model.parameter_count == 132_274_691
    assert model.parameter_ledger()["trainable_readout_parameters"] == model.parameter_count
