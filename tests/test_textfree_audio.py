import pytest

torch = pytest.importorskip("torch")


def _readout_class():
    from tiny_omni_decision.textfree_audio import TextFreeAudioReadout

    return TextFreeAudioReadout


CLASSES = ["yes", "no", "up", "down"]


def _options(classes):
    return [{"option_id": f"command-{name}", "class_id": name} for name in classes]


def test_textfree_audio_readout_emits_typed_probability_json():
    torch.manual_seed(17)
    model = _readout_class()(3, CLASSES).eval()
    result = model.decide(torch.tensor([0.2, -0.4, 0.8]), _options(CLASSES))
    assert result["status"] == "selected"
    assert len(result["options"]) == len(CLASSES)
    assert sum(item["probability"] for item in result["options"]) == pytest.approx(1.0)
    assert result["selected_option_id"].startswith("command-")


def test_textfree_audio_readout_is_option_order_equivariant():
    torch.manual_seed(17)
    model = _readout_class()(3, CLASSES).eval()
    feature = torch.tensor([0.2, -0.4, 0.8])
    forward = model.decide(feature, _options(CLASSES))
    order = ["up", "yes", "down", "no"]
    reversed_result = model.decide(feature, _options(order))
    p_forward = {item["option_id"]: item["probability"] for item in forward["options"]}
    p_reordered = {
        item["option_id"]: item["probability"] for item in reversed_result["options"]
    }
    assert p_forward == pytest.approx(p_reordered)
    assert forward["selected_option_id"] == reversed_result["selected_option_id"]


def test_textfree_audio_readout_supports_supplied_subset():
    model = _readout_class()(3, CLASSES).eval()
    result = model.decide(torch.zeros(3), _options(["up", "down"]))
    assert len(result["options"]) == 2
    assert sum(item["probability"] for item in result["options"]) == pytest.approx(1.0)


def test_textfree_audio_readout_save_reload_preserves_logits(tmp_path):
    torch.manual_seed(17)
    model = _readout_class()(3, CLASSES).eval()
    features = torch.randn(5, 3)
    expected = model(features)
    path = tmp_path / "readout.pt"
    torch.save(model.state_dict(), path)
    loaded = _readout_class()(3, CLASSES).eval()
    loaded.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    assert torch.equal(expected, loaded(features))


@pytest.mark.parametrize(
    "options",
    [
        _options(["yes"]),
        _options(["yes", "yes"]),
        _options(["yes", "unknown"]),
        [{"option_id": "same", "class_id": "yes"}, {"option_id": "same", "class_id": "no"}],
        [{"option_id": "yes", "class_id": "yes", "text": "should not be consumed"}],
    ],
)
def test_textfree_audio_readout_rejects_invalid_options(options):
    model = _readout_class()(3, CLASSES).eval()
    with pytest.raises(ValueError):
        model.decide(torch.zeros(3), options)


def test_textfree_audio_readout_rejects_invalid_feature_shapes_and_values():
    model = _readout_class()(3, CLASSES)
    with pytest.raises(ValueError, match="shape"):
        model(torch.zeros(1, 1, 3))
    with pytest.raises(ValueError, match="finite"):
        model(torch.tensor([0.0, float("nan"), 1.0]))
