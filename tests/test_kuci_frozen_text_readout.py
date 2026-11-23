import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from scripts.train_kuci_frozen_text_readout import (  # noqa: E402
    CandidateScorer,
    _cpu_state_dict,
    _line_dropped_layers,
    _masked_mean_pool,
    _metrics,
)


def test_shared_scorer_is_option_order_equivariant() -> None:
    torch.manual_seed(17)
    scorer = CandidateScorer(hidden_size=12).eval()
    features = torch.randn(5, 4, 12)
    permutation = [2, 0, 3, 1]

    with torch.inference_mode():
        original = scorer(features)
        reordered = scorer(features[:, permutation])

    torch.testing.assert_close(reordered, original[:, permutation], atol=1e-7, rtol=0)


def test_cpu_state_copy_does_not_move_or_mutate_module() -> None:
    scorer = CandidateScorer(hidden_size=12)
    before = {key: value.clone() for key, value in scorer.state_dict().items()}

    saved = _cpu_state_dict(scorer)

    assert all(value.device.type == "cpu" for value in saved.values())
    assert all(torch.equal(value, before[key]) for key, value in scorer.state_dict().items())
    assert all(value.device.type == "cpu" for value in scorer.parameters())


def test_masked_mean_pool_ignores_padding_and_averages_attended_tokens() -> None:
    hidden = torch.tensor([[[2.0, 4.0], [6.0, 8.0], [100.0, 100.0]]])
    attention_mask = torch.tensor([[1, 1, 0]])

    pooled = _masked_mean_pool(hidden, attention_mask)

    torch.testing.assert_close(pooled, torch.tensor([[4.0, 6.0]]))


def test_masked_mean_pool_rejects_mismatched_mask_shape() -> None:
    with pytest.raises(ValueError, match="masked mean pooling"):
        _masked_mean_pool(torch.ones(2, 3, 4), torch.ones(2, 2))


def test_metrics_return_finite_four_choice_scores() -> None:
    logits = torch.tensor([[4.0, 0.0, -1.0, -2.0], [0.0, 3.0, -1.0, -2.0]])
    targets = torch.tensor([0, 1])

    result = _metrics(logits, targets)

    assert result["count"] == 2
    assert result["accuracy"] == 1.0
    assert result["nll"] >= 0.0
    assert result["brier"] >= 0.0
    assert 0.0 <= result["ece_15_bins"] <= 1.0
    assert sum(result["ece_bin_counts"]) == 2


def test_metrics_rejects_no_rows_through_caller_contract() -> None:
    # Empty evaluation has no defined accuracy or ECE and must be screened upstream.
    with pytest.raises((ZeroDivisionError, RuntimeError, ValueError)):
        _metrics(torch.empty((0, 4)), torch.empty((0,), dtype=torch.long))


def test_line_truncation_is_exactly_the_two_unloaded_layers() -> None:
    keys = [
        "transformer.layer.{4, 5}.attention.q_lin.weight",
        "transformer.layer.{4, 5}.ffn.lin1.weight",
    ]

    assert _line_dropped_layers(keys) == {4, 5}
    assert _line_dropped_layers(["transformer.layer.3.attention.q_lin.weight"]) == {3}
