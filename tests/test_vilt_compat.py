import pytest

torch = pytest.importorskip("torch")


def test_only_expected_deterministic_buffer_is_removed():
    from tiny_omni_decision.vilt_compat import remove_legacy_position_ids

    key = "vilt.embeddings.text_embeddings.position_ids"
    weight = torch.randn(2, 3)
    original = {key: torch.arange(40).unsqueeze(0), "weight": weight}
    translated = remove_legacy_position_ids(original, 40)
    assert translated == {"weight": weight}
    assert key in original
    for values in [
        torch.arange(39).unsqueeze(0),
        torch.zeros(1, 40, dtype=torch.int64),
        torch.arange(40, dtype=torch.float32).unsqueeze(0),
    ]:
        with pytest.raises(ValueError):
            remove_legacy_position_ids({key: values}, 40)
