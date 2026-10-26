import pytest

pytest.importorskip("torch")
import torch

from tiny_omni_decision.temporal_fusion import TimestampedModalityFusion


def test_future_and_stale_tokens_are_unavailable_at_query_time():
    model = TimestampedModalityFusion({"audio": 2, "video": 3}, width=4)
    features = {
        "audio": torch.tensor([[[1.0, 0.0], [9.0, 9.0]]]),
        "video": torch.tensor([[[2.0, 0.0, 0.0]]]),
    }
    timestamps = {
        "audio": torch.tensor([[0.0, 11.0]]),
        "video": torch.tensor([[0.0]]),
    }
    available = {
        "audio": torch.tensor([[True, True]]),
        "video": torch.tensor([[True]]),
    }

    fused, weights, present = model(
        features,
        timestamps,
        available,
        torch.tensor([[10.0, 30.0]]),
        max_age_ms=15.0,
    )

    assert fused.shape == (1, 2, 4)
    assert weights.shape == (1, 2, 2)
    assert present["audio"].tolist() == [[True, False]]
    assert present["video"].tolist() == [[True, False]]
    assert torch.all(weights[0, 1] == 0)
    assert torch.all(fused[0, 1] == 0)


def test_future_token_content_cannot_change_causal_output():
    torch.manual_seed(17)
    model = TimestampedModalityFusion({"audio": 2}, width=4).eval()
    base = torch.tensor([[[0.25, -0.5], [2.0, 3.0]]])
    times = {"audio": torch.tensor([[0.0, 12.0]])}
    mask = {"audio": torch.tensor([[True, True]])}
    query = torch.tensor([[10.0]])
    first = model({"audio": base}, times, mask, query, max_age_ms=20.0)
    changed = base.clone()
    changed[0, 1] = torch.tensor([-100.0, 200.0])
    second = model({"audio": changed}, times, mask, query, max_age_ms=20.0)
    torch.testing.assert_close(first[0], second[0], rtol=0, atol=0)


def test_invalid_gap_does_not_shift_latest_valid_token_selection():
    torch.manual_seed(23)
    model = TimestampedModalityFusion({"sensor": 1}, width=4).eval()
    features = {"sensor": torch.tensor([[[1.0], [50.0], [3.0]]])}
    times = {"sensor": torch.tensor([[0.0, 5.0, 10.0]])}
    mask = {"sensor": torch.tensor([[True, False, True]])}
    query = torch.tensor([[10.0]])
    first = model(features, times, mask, query, max_age_ms=10.0)
    features["sensor"][0, 1, 0] = -100.0
    second = model(features, times, mask, query, max_age_ms=10.0)
    torch.testing.assert_close(first[0], second[0], rtol=0, atol=0)


def test_invalid_shapes_and_nonmonotonic_timestamps_are_rejected():
    model = TimestampedModalityFusion({"sensor": 2}, width=4)
    features = {"sensor": torch.zeros(1, 2, 2)}
    masks = {"sensor": torch.ones(1, 2, dtype=torch.bool)}
    with pytest.raises(ValueError, match="strictly increase"):
        model(features, {"sensor": torch.tensor([[2.0, 1.0]])}, masks,
              torch.tensor([[2.0]]), max_age_ms=5.0)


def test_concat_control_uses_same_causal_alignment_and_masks_as_gated_readout():
    torch.manual_seed(31)
    gated_model = TimestampedModalityFusion({"audio": 2, "video": 1}, width=4)
    features = {
        "audio": torch.tensor([[[1.0, 0.0], [9.0, 9.0]]]),
        "video": torch.tensor([[[2.0], [3.0]]]),
    }
    timestamps = {
        "audio": torch.tensor([[0.0, 11.0]]),
        "video": torch.tensor([[0.0, 31.0]]),
    }
    masks = {
        "audio": torch.tensor([[True, True]]),
        "video": torch.tensor([[True, True]]),
    }
    queries = torch.tensor([[10.0, 30.0]])

    gated, _, gated_masks = gated_model(
        features, timestamps, masks, queries, max_age_ms=15.0
    )
    torch.manual_seed(31)
    concat_model = TimestampedModalityFusion(
        {"audio": 2, "video": 1}, width=4, readout_policy="concat_mlp"
    )
    assert not hasattr(gated_model, "concat_mlp")
    assert not hasattr(concat_model, "gate")
    concat, concat_masks = concat_model.forward_concat(
        features, timestamps, masks, queries, max_age_ms=15.0
    )

    assert concat.shape == gated.shape == (1, 2, 4)
    assert concat_masks.keys() == gated_masks.keys()
    for name in gated_masks:
        assert torch.equal(concat_masks[name], gated_masks[name])
    assert concat_masks["audio"].tolist() == [[True, False]]
    assert concat_masks["video"].tolist() == [[True, False]]
    assert torch.isfinite(concat).all()
    assert torch.equal(concat[:, 1], torch.zeros_like(concat[:, 1]))


def test_concat_control_is_zero_when_all_modalities_are_missing_or_stale():
    model = TimestampedModalityFusion(
        {"sensor": 2}, width=3, readout_policy="concat_mlp"
    )
    output, masks = model.forward_concat(
        {"sensor": torch.tensor([[[1.0, 2.0]]])},
        {"sensor": torch.tensor([[0.0]])},
        {"sensor": torch.tensor([[False]])},
        torch.tensor([[10.0]]),
        max_age_ms=5.0,
    )

    assert masks["sensor"].tolist() == [[False]]
    assert torch.equal(output, torch.zeros_like(output))
