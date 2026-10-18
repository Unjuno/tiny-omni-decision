import pytest

torch = pytest.importorskip("torch")  # noqa: F401
pytest.importorskip("transformers")

from scripts.compare_frozen_text_pooling import VARIANTS, _pool_hidden  # noqa: E402


def test_text_pooling_variants_are_distinct_and_mask_padding():
    hidden = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0], [99.0, 99.0]], [[2.0, 1.0], [5.0, 7.0], [8.0, 9.0]]]
    )
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    final_mean = _pool_hidden(hidden, mask, "final_layer_masked_mean")
    first = _pool_hidden(hidden, mask, "final_layer_first_token")
    middle = _pool_hidden(hidden + 10, mask, "middle_layer_masked_mean")

    assert VARIANTS == (
        "final_layer_masked_mean",
        "middle_layer_masked_mean",
        "final_layer_first_token",
    )
    assert torch.equal(final_mean, torch.tensor([[2.0, 3.0], [5.0, 17.0 / 3.0]]))
    assert torch.equal(first, torch.tensor([[1.0, 2.0], [2.0, 1.0]]))
    assert torch.allclose(middle, final_mean + 10)


def test_text_pooling_rejects_shape_and_variant_errors():
    hidden = torch.zeros((1, 2, 3))
    with pytest.raises(ValueError, match="hidden must"):
        _pool_hidden(hidden, torch.ones((1, 3)), "final_layer_masked_mean")
    with pytest.raises(ValueError, match="unknown pooling"):
        _pool_hidden(hidden, torch.ones((1, 2)), "learned_attention")
