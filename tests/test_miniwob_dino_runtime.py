import pytest

torch = pytest.importorskip("torch")
from PIL import Image  # noqa: E402

from tiny_omni_decision.miniwob_dino_readout import MiniWoBOptionScorer  # noqa: E402
from tiny_omni_decision.miniwob_dino_runtime import MiniWoBDinoDecisionRuntime  # noqa: E402


class _FakeDino(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)

    def forward(self, *, pixel_values):
        batch = pixel_values.shape[0]
        tokens = torch.arange(257 * 384, dtype=pixel_values.dtype, device=pixel_values.device)
        tokens = tokens.reshape(1, 257, 384).expand(batch, -1, -1)
        return type("Output", (), {"last_hidden_state": tokens + self.anchor})()


def _options():
    return [
        {"id": f"a{i}", "text": f"Candidate {i}", "x_norm": x, "y_norm": y}
        for i, (x, y) in enumerate(((0.1, 0.2), (0.4, 0.3), (0.6, 0.7), (0.9, 0.8)))
    ]


def test_json_runtime_returns_normalized_option_probabilities_and_is_permutation_equivariant():
    torch.manual_seed(17)
    runtime = MiniWoBDinoDecisionRuntime(_FakeDino(), MiniWoBOptionScorer().eval())
    image = Image.new("RGB", (160, 210), (32, 64, 128))
    options = _options()
    first = runtime.predict(request_id="one", image=image, options=options)
    order = [2, 0, 3, 1]
    second = runtime.predict(
        request_id="two", image=image, options=[options[index] for index in order]
    )
    first_by_id = {item["id"]: item["probability"] for item in first["options"]}
    second_by_id = {item["id"]: item["probability"] for item in second["options"]}
    assert first["selected_option_id"] in first_by_id
    assert sum(first_by_id.values()) == pytest.approx(1.0, abs=1e-6)
    assert first_by_id == pytest.approx(second_by_id, abs=1e-6)
    assert first["decision_type"] == "choice"


def test_json_runtime_fails_closed_on_duplicate_ids_and_wrong_screen_shape():
    runtime = MiniWoBDinoDecisionRuntime(_FakeDino(), MiniWoBOptionScorer().eval())
    options = _options()
    options[1]["id"] = options[0]["id"]
    with pytest.raises(ValueError, match="unique"):
        runtime.predict(request_id="duplicate", image=Image.new("RGB", (160, 210)), options=options)
    with pytest.raises(ValueError, match="160x210"):
        runtime.predict(
            request_id="wrong-shape",
            image=Image.new("RGB", (224, 224)),
            options=_options(),
        )
