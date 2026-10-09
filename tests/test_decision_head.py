import pytest
import torch

from tiny_omni_decision.decision_head import (
    OptionDecisionHead,
    decision_head_parameter_count,
    train_decision_head_one_pass,
)


def test_head_has_expected_parameter_count_and_variable_choice_count():
    head = OptionDecisionHead()
    assert decision_head_parameter_count(head) == 295_169
    query = torch.randn(768)
    assert head(query, torch.randn(2, 768)).shape == (2,)
    assert head(query, torch.randn(7, 768)).shape == (7,)


def test_head_scores_options_equivariantly_when_choice_order_changes():
    torch.manual_seed(17)
    head = OptionDecisionHead(embedding_dim=8, hidden_dim=4)
    query = torch.randn(8)
    options = torch.randn(4, 8)
    permutation = torch.tensor([2, 0, 3, 1])
    assert torch.allclose(head(query, options[permutation]), head(query, options)[permutation])


def test_head_rejects_invalid_embedding_shapes():
    head = OptionDecisionHead(embedding_dim=8, hidden_dim=4)
    with pytest.raises(ValueError, match="query must"):
        head(torch.randn(1, 8), torch.randn(2, 8))
    with pytest.raises(ValueError, match="options must"):
        head(torch.randn(8), torch.randn(1, 8))


def test_one_pass_updates_only_head_once_per_ordered_example():
    torch.manual_seed(17)
    head = OptionDecisionHead(embedding_dim=8, hidden_dim=4)
    before = [parameter.detach().clone() for parameter in head.parameters()]
    queries = torch.randn(3, 8)
    options = torch.randn(7, 8)
    offsets = torch.tensor([0, 2, 5, 7])
    targets = torch.tensor([1, 0, 1])
    teacher = torch.tensor([0.2, 0.8, 0.6, 0.3, 0.1, 0.4, 0.6])
    result = train_decision_head_one_pass(
        head,
        queries,
        options,
        offsets,
        targets,
        teacher,
        learning_rate=3e-4,
        weight_decay=0.01,
        option_kl_weight=1.0,
        cross_entropy_weight=0.2,
        brier_weight=0.2,
    )
    assert result["examples_consumed"] == 3
    assert result["optimizer_updates"] == 3
    assert [row["update"] for row in result["history"]] == [1, 2, 3]
    assert all(
        not torch.equal(old, new) for old, new in zip(before, head.parameters(), strict=True)
    )
