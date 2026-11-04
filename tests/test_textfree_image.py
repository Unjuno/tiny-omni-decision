import pytest

torch = pytest.importorskip("torch")

def test_image_readout_maps_supplied_options_to_json_probabilities():
    model = _readout_type()(2, {"color": ["blue", "red", "yellow"]})
    with torch.no_grad():
        model.classifiers["color"].weight.copy_(torch.tensor([[1.0, 0.0], [0, 1], [-1, 0]]))
        model.classifiers["color"].bias.zero_()

    result = model.decide(
        torch.tensor([2.0, 1.0]),
        "color",
        [
            {"option_id": "o-yellow", "class_id": "yellow"},
            {"option_id": "o-blue", "class_id": "blue"},
            {"option_id": "o-red", "class_id": "red"},
        ],
    )

    assert result["selected_option_id"] == "o-blue"
    assert [option["option_id"] for option in result["options"]] == [
        "o-yellow",
        "o-blue",
        "o-red",
    ]
    assert sum(option["probability"] for option in result["options"]) == pytest.approx(1.0)


def test_image_readout_rejects_incomplete_or_duplicate_options():
    model = _readout_type()(2, {"color": ["blue", "red", "yellow"]})
    features = torch.ones(2)

    with pytest.raises(ValueError, match="at least two"):
        model.decide(features, "color", [{"option_id": "blue", "class_id": "blue"}])
    with pytest.raises(ValueError, match="unique option IDs"):
        model.decide(
            features,
            "color",
            [
                {"option_id": "same", "class_id": "blue"},
                {"option_id": "same", "class_id": "red"},
                {"option_id": "yellow", "class_id": "yellow"},
            ],
        )
    with pytest.raises(ValueError, match="unique known class IDs"):
        model.decide(
            features,
            "color",
            [
                {"option_id": "blue-1", "class_id": "blue"},
                {"option_id": "blue-2", "class_id": "blue"},
            ],
        )


def test_image_readout_renormalizes_over_supplied_option_subset():
    model = _readout_type()(2, {"color": ["blue", "red", "yellow"]})
    result = model.decide(
        torch.tensor([0.5, -0.25]),
        "color",
        [
            {"option_id": "red-option", "class_id": "red"},
            {"option_id": "yellow-option", "class_id": "yellow"},
        ],
    )

    assert len(result["options"]) == 2
    assert sum(option["probability"] for option in result["options"]) == pytest.approx(1.0)


def test_four_taxonomy_readout_has_a_measured_small_parameter_budget(tmp_path):
    from tiny_omni_decision.dataset import CLEVR4_TAXONOMIES

    model_type = _readout_type()
    model = model_type(768, CLEVR4_TAXONOMIES).eval()
    assert sum(parameter.numel() for parameter in model.parameters()) == 30_760

    features = torch.randn(4, 768)
    expected = {name: model(features, name) for name in CLEVR4_TAXONOMIES}
    path = tmp_path / "readout.pt"
    torch.save(model.state_dict(), path)
    reloaded = model_type(768, CLEVR4_TAXONOMIES).eval()
    reloaded.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    assert all(torch.equal(expected[name], reloaded(features, name)) for name in expected)


def _readout_type():
    from tiny_omni_decision.textfree_image import TextFreeImageReadout

    return TextFreeImageReadout


def test_deterministic_asset_split_is_id_based_and_disjoint():
    from tiny_omni_decision.textfree_image import deterministic_asset_split

    ids = [f"image-{index}" for index in range(100)]
    first_train, first_validation = deterministic_asset_split(ids, seed=17, validation_fraction=0.2)
    second_train, second_validation = deterministic_asset_split(
        ids, seed=17, validation_fraction=0.2
    )

    assert first_train == second_train
    assert first_validation == second_validation
    assert not set(first_train).intersection(first_validation)
    assert set(first_train).union(first_validation) == set(range(len(ids)))
    reordered_ids = list(reversed(ids))
    reordered_train, reordered_validation = deterministic_asset_split(
        reordered_ids, seed=17, validation_fraction=0.2
    )
    train_ids = {reordered_ids[index] for index in reordered_train}
    validation_ids = {reordered_ids[index] for index in reordered_validation}
    assert train_ids == {ids[index] for index in first_train}
    assert validation_ids == {ids[index] for index in first_validation}
