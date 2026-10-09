import pytest

from tiny_omni_decision.ternary_diagnostic_utils import component_for_target


@pytest.mark.parametrize(
    ("name", "component"),
    [
        ("audio_tower.layers.0.proj.weight", "audio_path"),
        ("embed_audio.projection.weight", "audio_path"),
        ("vision_tower.layers.0.proj.weight", "vision_path"),
        ("embed_vision.projection.weight", "vision_path"),
        ("language_model.layers.0.self_attn.q_proj.weight", "shared_decoder"),
    ],
)
def test_component_mapping_covers_encoder_projection_and_decoder_paths(name, component):
    assert component_for_target(name) == component


def test_component_mapping_fails_closed_for_unknown_root():
    with pytest.raises(ValueError, match="unclassified ternary target root"):
        component_for_target("projector.weight")
