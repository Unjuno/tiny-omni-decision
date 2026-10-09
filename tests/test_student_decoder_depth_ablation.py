import pytest

from tiny_omni_decision.ternary_diagnostic_utils import group_decoder_layer_targets


def test_decoder_targets_are_split_into_three_contiguous_equal_groups():
    targets = tuple(
        f"language_model.layers.{layer}.self_attn.{projection}.weight"
        for layer in range(24)
        for projection in ("q_proj", "k_proj", "v_proj")
    ) + ("language_model.embed_tokens.weight",)

    groups = group_decoder_layer_targets(targets)

    assert tuple(groups) == ("early", "middle", "late")
    assert len(groups["early"]) == len(groups["middle"]) == len(groups["late"]) == 24
    assert {name.split(".")[2] for name in groups["early"]} == {str(layer) for layer in range(8)}
    assert {name.split(".")[2] for name in groups["middle"]} == {
        str(layer) for layer in range(8, 16)
    }
    assert {name.split(".")[2] for name in groups["late"]} == {
        str(layer) for layer in range(16, 24)
    }
    assert "language_model.embed_tokens.weight" not in {
        name for group in groups.values() for name in group
    }


def test_decoder_depth_grouping_rejects_missing_layer_ids():
    targets = (
        "language_model.layers.0.self_attn.q_proj.weight",
        "language_model.layers.2.self_attn.q_proj.weight",
    )

    with pytest.raises(ValueError, match="contiguous"):
        group_decoder_layer_targets(targets)


def test_decoder_depth_grouping_rejects_layer_count_not_divisible_by_three():
    targets = tuple(f"language_model.layers.{layer}.self_attn.q_proj.weight" for layer in range(4))

    with pytest.raises(ValueError, match="three equal groups"):
        group_decoder_layer_targets(targets)
