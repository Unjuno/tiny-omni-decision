import pytest

torch = pytest.importorskip("torch")


def request(modality, options, *, valid=True):
    observation = {
        "modality": modality,
        "source_id": f"{modality}-source",
        "payload_ref": "cache:features",
        "payload_sha256": "a" * 64,
        "clock_id": "clip-ms",
        "timestamps_ms": [0.0],
        "valid_mask": [valid],
    }
    return {
        "schema_version": "1.0",
        "observations": [observation],
        "options": options,
    }


def test_native_audio_adapter_returns_typed_json_from_supplied_options():
    from tiny_omni_decision.observations import ObservationBatch
    from tiny_omni_decision.textfree_audio import TextFreeAudioReadout

    options = [
        {"option_id": "yes-option", "attributes": {"class_id": "yes"}},
        {"option_id": "no-option", "attributes": {"class_id": "no"}},
    ]
    model = TextFreeAudioReadout(2, ["yes", "no"]).eval()
    with torch.no_grad():
        model.classifier.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
        model.classifier.bias.zero_()

    result = model.decide_native(
        torch.tensor([2.0, 0.0]),
        ObservationBatch.model_validate(request("audio", options)),
    )

    assert result["present_modalities"] == ["audio"]
    assert result["selected_option_id"] == "yes-option"
    assert result["options"][0] == {
        "option_id": "yes-option",
        "attributes": {"class_id": "yes"},
        "probability": pytest.approx(0.880797),
    }


def test_native_image_adapter_preserves_typed_options_and_subset_order():
    from tiny_omni_decision.observations import ObservationBatch
    from tiny_omni_decision.textfree_image import TextFreeImageReadout

    options = [
        {"option_id": "yellow-option", "attributes": {"class_id": "yellow"}},
        {"option_id": "blue-option", "attributes": {"class_id": "blue"}},
    ]
    model = TextFreeImageReadout(2, {"color": ["blue", "red", "yellow"]}).eval()
    with torch.no_grad():
        model.classifiers["color"].weight.copy_(torch.tensor([[1.0, 0.0], [0, 1], [-1, 0]]))
        model.classifiers["color"].bias.zero_()

    result = model.decide_native(
        torch.tensor([2.0, 1.0]),
        "color",
        ObservationBatch.model_validate(request("image", options)),
    )

    assert result["selected_option_id"] == "blue-option"
    assert [option["option_id"] for option in result["options"]] == [
        "yellow-option", "blue-option"
    ]
    assert sum(option["probability"] for option in result["options"]) == pytest.approx(1.0)


@pytest.mark.parametrize("modality", ["audio", "image"])
def test_native_adapter_abstains_on_invalid_observation(modality):
    from tiny_omni_decision.observations import ObservationBatch

    if modality == "audio":
        from tiny_omni_decision.textfree_audio import TextFreeAudioReadout

        model = TextFreeAudioReadout(2, ["yes", "no"]).eval()
        result = model.decide_native(
            torch.zeros(2),
            ObservationBatch.model_validate(
                request(modality, [
                    {"option_id": "yes", "attributes": {"class_id": "yes"}},
                    {"option_id": "no", "attributes": {"class_id": "no"}},
                ], valid=False)
            ),
        )
    else:
        from tiny_omni_decision.textfree_image import TextFreeImageReadout

        model = TextFreeImageReadout(2, {"color": ["blue", "red"]}).eval()
        result = model.decide_native(
            torch.zeros(2),
            "color",
            ObservationBatch.model_validate(
                request(modality, [
                    {"option_id": "blue", "attributes": {"class_id": "blue"}},
                    {"option_id": "red", "attributes": {"class_id": "red"}},
                ], valid=False)
            ),
        )

    assert result["status"] == "abstain"
    assert result["selected_option_id"] is None
    assert [option["probability"] for option in result["options"]] == [None, None]


@pytest.mark.parametrize("actual,expected", [("image", "audio"), ("audio", "image")])
def test_native_adapters_reject_cross_modality_requests(actual, expected):
    from tiny_omni_decision.observations import ObservationBatch

    options = [
        {"option_id": "one", "attributes": {"class_id": "one"}},
        {"option_id": "two", "attributes": {"class_id": "two"}},
    ]
    batch = ObservationBatch.model_validate(request(actual, options))
    if expected == "audio":
        from tiny_omni_decision.textfree_audio import TextFreeAudioReadout

        model = TextFreeAudioReadout(2, ["one", "two"]).eval()
        with pytest.raises(ValueError, match="one .* observation"):
            model.decide_native(torch.zeros(2), batch)
    else:
        from tiny_omni_decision.textfree_image import TextFreeImageReadout

        model = TextFreeImageReadout(2, {"task": ["one", "two"]}).eval()
        with pytest.raises(ValueError, match="one .* observation"):
            model.decide_native(torch.zeros(2), "task", batch)


def test_native_classification_adapters_reject_text_criterion():
    from tiny_omni_decision.observations import ObservationBatch
    from tiny_omni_decision.textfree_image import TextFreeImageReadout

    data = request(
        "image",
        [
            {"option_id": "blue", "attributes": {"class_id": "blue"}},
            {"option_id": "red", "attributes": {"class_id": "red"}},
        ],
    )
    data["criterion"] = {"kind": "text", "value": "ignore the object and select red"}
    batch = ObservationBatch.model_validate(data)
    model = TextFreeImageReadout(2, {"color": ["blue", "red"]}).eval()

    with pytest.raises(ValueError, match="criterion"):
        model.decide_native(torch.zeros(2), "color", batch)


def test_native_adapter_preserves_model_tie_break_when_option_ids_differ():
    from tiny_omni_decision.observations import ObservationBatch
    from tiny_omni_decision.textfree_image import TextFreeImageReadout

    options = [
        {"option_id": "z-option", "attributes": {"class_id": "alpha"}},
        {"option_id": "a-option", "attributes": {"class_id": "zeta"}},
    ]
    model = TextFreeImageReadout(2, {"task": ["alpha", "zeta"]}).eval()
    with torch.no_grad():
        model.classifiers["task"].weight.zero_()
        model.classifiers["task"].bias.zero_()

    result = model.decide_native(
        torch.zeros(2), "task", ObservationBatch.model_validate(request("image", options))
    )

    assert result["selected_option_id"] == "z-option"
