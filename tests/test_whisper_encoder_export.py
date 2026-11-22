import pytest
import torch

from scripts.audit_whisper_encoder_only_export import extract_encoder_state


def test_extract_encoder_state_strips_only_the_encoder_prefix() -> None:
    encoder_weight = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    state = {
        "model.encoder.layers.0.q_proj.weight": encoder_weight,
        "model.decoder.layers.0.q_proj.weight": torch.ones(2, 3),
    }

    extracted = extract_encoder_state(state)

    assert list(extracted) == ["layers.0.q_proj.weight"]
    torch.testing.assert_close(extracted["layers.0.q_proj.weight"], encoder_weight, rtol=0, atol=0)


def test_extract_encoder_state_fails_closed_when_prefix_is_wrong() -> None:
    with pytest.raises(ValueError, match="no encoder tensors"):
        extract_encoder_state({"model.decoder.weight": torch.ones(1)})


def test_extract_encoder_state_requires_a_dotted_namespace() -> None:
    with pytest.raises(ValueError, match="dotted namespace"):
        extract_encoder_state({"encoder.weight": torch.ones(1)}, prefix="encoder")
