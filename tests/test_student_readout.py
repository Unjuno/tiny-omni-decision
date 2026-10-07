from __future__ import annotations

import pytest


@pytest.fixture
def readout():
    torch = pytest.importorskip("torch")
    from tiny_omni_decision.student import mean_pool_projected_tokens, supplied_option_logits

    return torch, mean_pool_projected_tokens, supplied_option_logits


def test_mean_pool_matches_attention_mask_and_keeps_prompt_tokens(readout) -> None:
    torch, mean_pool_projected_tokens, _ = readout
    embeddings = torch.tensor([[[1.0, 3.0], [3.0, 5.0], [99.0, 99.0]]])
    mask = torch.tensor([[1, 1, 0]])

    pooled = mean_pool_projected_tokens(embeddings, mask)

    torch.testing.assert_close(pooled, torch.tensor([[2.0, 4.0]]))


def test_decision_query_and_options_use_the_same_similarity_prefix() -> None:
    from tiny_omni_decision.student import decision_option_text, decision_query_text

    query = decision_query_text(
        "A red cube is left of a sphere.", "Which is farther?", media_token="<|video|>"
    )
    option = decision_option_text("The cube.")

    assert query == (
        "task: sentence similarity | query: A red cube is left of a sphere.\n"
        "Which is farther?\n<|video|>"
    )
    assert option == "task: sentence similarity | query: The cube."
    assert "The cube" not in query


def test_supplied_option_logits_preserve_order_and_backpropagate(readout) -> None:
    torch, _, supplied_option_logits = readout
    query = torch.tensor([1.0, 0.0], requires_grad=True)
    options = torch.tensor(
        [[0.0, 1.0], [1.0, 0.0], [-1.0, 0.0]], requires_grad=True
    )

    logits = supplied_option_logits(query, options, temperature=0.5)
    probabilities = torch.softmax(logits, dim=-1)
    loss = -torch.log(probabilities[1])
    loss.backward()

    torch.testing.assert_close(logits, torch.tensor([0.0, 2.0, -2.0]))
    assert probabilities.argmax().item() == 1
    assert probabilities.sum().item() == pytest.approx(1.0)
    assert query.grad is not None and torch.isfinite(query.grad).all()
    assert options.grad is not None and torch.isfinite(options.grad).all()
    assert query.grad.abs().sum().item() > 0
    assert options.grad.abs().sum().item() > 0


def test_mean_pool_rejects_empty_attention_mask(readout) -> None:
    torch, mean_pool_projected_tokens, _ = readout
    with pytest.raises(ValueError, match="at least one attended token"):
        mean_pool_projected_tokens(torch.ones(1, 2, 3), torch.zeros(1, 2))


def test_option_readout_rejects_invalid_temperature_and_dimensions(readout) -> None:
    torch, _, supplied_option_logits = readout
    query = torch.ones(3)
    options = torch.ones(2, 3)
    with pytest.raises(ValueError, match="temperature"):
        supplied_option_logits(query, options, temperature=0)
    with pytest.raises(ValueError, match="dimensions"):
        supplied_option_logits(query, torch.ones(2, 4), temperature=1.0)


def test_option_readout_rejects_zero_embeddings(readout) -> None:
    torch, _, supplied_option_logits = readout
    with pytest.raises(ValueError, match="nonzero"):
        supplied_option_logits(torch.zeros(3), torch.ones(2, 3), temperature=1.0)
