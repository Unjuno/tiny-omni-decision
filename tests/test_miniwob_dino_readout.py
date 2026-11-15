import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from tiny_omni_decision.miniwob_dino_readout import (  # noqa: E402
    MiniWoBOptionScorer,
    classification_metrics,
    deranged_indices,
    permute_candidate_axis,
)
from tiny_omni_decision.miniwob_dino_runtime import MiniWoBDinoRuntime  # noqa: E402


def test_candidate_scorer_is_permutation_equivariant():
    torch.manual_seed(17)
    model = MiniWoBOptionScorer().eval()
    features = torch.randn(3, 4, 1154)
    permutation = torch.tensor([2, 0, 3, 1])
    reordered, inverse = permute_candidate_axis(features, permutation)
    with torch.inference_mode():
        expected = model(features).index_select(1, permutation)
        actual = model(reordered).index_select(1, inverse)
    torch.testing.assert_close(actual, model(features), rtol=0, atol=0)
    assert torch.equal(expected.index_select(1, inverse), model(features))


def test_metrics_match_four_choice_reference_values():
    probabilities = torch.tensor([[0.7, 0.1, 0.1, 0.1], [0.25, 0.25, 0.25, 0.25]])
    targets = torch.tensor([0, 1])
    metrics = classification_metrics(probabilities, targets)
    assert metrics["count"] == 2
    assert metrics["accuracy"] == 0.5
    assert metrics["brier"] == pytest.approx((0.12 + 0.75) / 2)
    expected_nll = (-torch.log(torch.tensor(0.7)) + torch.log(torch.tensor(4.0))) / 2
    assert metrics["nll"] == pytest.approx(expected_nll.item())


def test_scorer_rejects_wrong_shapes_and_nonfinite_inputs():
    model = MiniWoBOptionScorer()
    with pytest.raises(ValueError):
        model(torch.zeros(2, 1154))
    invalid = torch.zeros(2, 4, 1154)
    invalid[0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        model(invalid)


def test_metric_validation_rejects_non_normalized_probabilities():
    with pytest.raises(ValueError):
        classification_metrics(torch.tensor([[0.5, 0.2, 0.2, 0.2]]), torch.tensor([0]))


def test_screen_derangement_is_deterministic_complete_and_has_no_fixed_points():
    first = deranged_indices(16, seed=17)
    assert first == deranged_indices(16, seed=17)
    assert sorted(first) == list(range(16))
    assert all(index != donor for index, donor in enumerate(first))
    with pytest.raises(ValueError):
        deranged_indices(1, seed=17)


def test_decide_json_preserves_option_identity_order_and_permutation():
    torch.manual_seed(17)
    model = MiniWoBOptionScorer().eval()
    features = torch.randn(1, 4, 1154)
    option_ids = ["red", "blue", "green", "yellow"]
    result = model.decide_json(features, option_ids)
    reversed_result = model.decide_json(features.flip(1), option_ids[::-1])
    assert result["schema_version"] == "1.0"
    assert [item["option_id"] for item in result["options"]] == option_ids
    assert result["selected_option_id"] == reversed_result["selected_option_id"]
    assert {
        item["option_id"]: item["probability"] for item in result["options"]
    } == {
        item["option_id"]: item["probability"] for item in reversed_result["options"]
    }


@pytest.mark.parametrize(
    ("option_ids", "temperature"),
    [(["same", "same"], 1.0), (["a", "b"], 0.0), (["a", "b"], float("nan"))],
)
def test_decide_json_rejects_invalid_ids_and_temperature(option_ids, temperature):
    with pytest.raises(ValueError):
        MiniWoBOptionScorer().decide_json(
            torch.zeros(1, 2, 1154), option_ids, temperature=temperature
        )


def test_decide_json_rejects_non_string_option_ids():
    with pytest.raises(ValueError):
        MiniWoBOptionScorer().decide_json(torch.zeros(1, 2, 1154), [{}, "valid"])


class _FakeDino(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)

    def forward(self, pixel_values):
        batch = pixel_values.shape[0]
        token_values = torch.arange(257 * 384, dtype=torch.float32).reshape(1, 257, 384)
        tokens = token_values.expand(batch, -1, -1) + pixel_values.mean() + self.anchor
        return type("Output", (), {"last_hidden_state": tokens})()


def test_frozen_dino_runtime_encodes_coordinates_and_returns_json():
    runtime = MiniWoBDinoRuntime(_FakeDino(), MiniWoBOptionScorer())
    image = Image.new("RGB", (160, 210), "white")
    coordinates = [(0.2, 0.3), (0.8, 0.7), (0.4, 0.5)]
    features = runtime.encode_options(image, coordinates)
    assert features.shape == (3, 1154)
    torch.testing.assert_close(features[:, -2:], torch.tensor(coordinates))
    batched = runtime.encode_options_batch([image, image], [coordinates, coordinates])
    torch.testing.assert_close(batched[0], features)
    torch.testing.assert_close(batched[1], features)
    result = runtime.decide(image, ["left", "right", "middle"], coordinates)
    assert result["schema_version"] == "1.0"
    assert [option["option_id"] for option in result["options"]] == [
        "left",
        "right",
        "middle",
    ]


def test_frozen_dino_runtime_rejects_out_of_range_coordinates():
    runtime = MiniWoBDinoRuntime(_FakeDino(), MiniWoBOptionScorer())
    image = Image.new("RGB", (160, 210), "white")
    with pytest.raises(ValueError):
        runtime.encode_options(image, [(0.1, 0.2), (1.1, 0.3)])
