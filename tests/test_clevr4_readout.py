import pytest
import torch

from tiny_omni_decision.clevr4_readout import build_candidate_features, classification_quality


def test_candidate_features_reuse_image_for_supplied_options():
    image = torch.arange(2 * 768, dtype=torch.float32).reshape(2, 768)
    options = {"color": torch.ones(10, 384), "shape": torch.full((10, 384), 2.0)}

    features = build_candidate_features(image, options, ["color", "shape"])

    assert features.shape == (2, 2, 10, 1152)
    torch.testing.assert_close(features[1, 0, 4, :768], image[1])
    torch.testing.assert_close(features[0, 0, 4, 768:], options["color"][4])
    torch.testing.assert_close(features[0, 1, 4, 768:], options["shape"][4])


def test_candidate_features_reject_unequal_option_counts():
    with pytest.raises(ValueError, match="equal option counts"):
        build_candidate_features(
            torch.zeros(1, 768),
            {"a": torch.zeros(2, 384), "b": torch.zeros(3, 384)},
            ["a", "b"],
        )


def test_quality_metrics_match_known_probabilities():
    probabilities = torch.tensor([[0.8, 0.2], [0.4, 0.6]])
    targets = torch.tensor([0, 0])

    metrics = classification_quality(probabilities, targets, bins=2)

    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["nll"] == pytest.approx(
        (-torch.log(torch.tensor(0.8)) - torch.log(torch.tensor(0.4))).item() / 2
    )
    assert metrics["brier"] == pytest.approx(
        ((0.8 - 1) ** 2 + (0.2) ** 2 + (0.4 - 1) ** 2 + 0.6**2) / 2
    )
    assert metrics["examples"] == 2


def test_quality_metrics_reject_non_normalized_probabilities():
    with pytest.raises(ValueError, match="sum to one"):
        classification_quality(torch.tensor([[0.8, 0.8]]), torch.tensor([0]))
