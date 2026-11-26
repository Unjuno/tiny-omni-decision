import numpy as np
import pytest


def test_decomposed_path_matches_native_head_with_missing_modality():
    torch = pytest.importorskip("torch")
    from scripts.lombard_temporal_model import TemporalSlotHead
    from scripts.lombard_visual_state import latent_states, readout_from_latent

    torch.manual_seed(17)
    model = TemporalSlotHead().eval()
    audio, video = torch.randn(3, 384), torch.randn(3, 16, 768)
    mask = torch.tensor([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    with torch.no_grad():
        expected = model(audio, video, mask).numpy()
    actual = readout_from_latent(model, audio, latent_states(model, video), mask)
    np.testing.assert_array_equal(actual, expected)
    assert all(p.requires_grad and p.grad is None for p in model.parameters())


def test_constants_are_train_derived_and_mask_pair_has_equal_visual_state():
    torch = pytest.importorskip("torch")
    from scripts.lombard_visual_state import latent_controls

    train = torch.arange(4 * 64).reshape(4, 64).float()
    mask = torch.ones(3, 2)
    result = latent_controls(train, mask)
    torch.testing.assert_close(
        result["latent_train_mean"][0], train.mean(0).repeat(3, 1), rtol=0, atol=0
    )
    torch.testing.assert_close(
        result["latent_train_first"][0], train[:1].repeat(3, 1), rtol=0, atol=0
    )
    assert result["latent_zero_present"][0].eq(0).all()
    assert result["latent_zero_absent"][0].eq(0).all()
    assert result["latent_zero_present"][1].eq(1).all()
    assert result["latent_zero_absent"][1][:, 1].eq(0).all()
    assert mask.eq(1).all()


def test_invalid_latents_and_masks_fail_closed():
    torch = pytest.importorskip("torch")
    from scripts.lombard_visual_state import latent_controls

    with pytest.raises(ValueError):
        latent_controls(torch.ones(3, 63), torch.ones(3, 2))
    with pytest.raises(ValueError):
        latent_controls(torch.ones(3, 64), torch.full((3, 2), 0.5))
