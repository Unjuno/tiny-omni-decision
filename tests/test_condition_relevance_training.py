"""Real losses and selection inputs for a matched, unmasked comparison."""

import math

import numpy as np
import pytest

from scripts.condition_relevance_training import (
    accounting,
    balanced_relevance_loss,
    bootstrap_relevance,
    forward_with_relevance,
    load_trainable,
    metric_summary,
    row_key,
    sample_order,
    save_trainable,
    training_objective,
)


def example(i, field="count", group=None):
    return dict(
        id=str(i),
        group=str(i if group is None else group),
        split="train",
        source="condition_relevance_v1",
        field=field,
        query_index=8,
        option_form="sentence",
        options=["1", "2"],
        target="1",
        state='{"count":"1"}',
        question="How many?",
    )


def test_order_exhausts_three_pools_once_with_same_schedule_for_both_arms():
    pools = [[example(f"{p}:{i}") for i in range(n)] for p, n in enumerate((4, 4, 2))]
    order = sample_order(pools, sizes=(2, 2, 1))
    assert order == sample_order(pools, sizes=(2, 2, 1))
    assert len(order) == 2
    assert sorted(tuple(v) for b in order for v in b) == [
        (p, i) for p, n in enumerate((4, 4, 2)) for i in range(n)
    ]
    assert all([sum(p == k for p, _ in b) for k in range(3)] == [2, 2, 1] for b in order)
    with pytest.raises(ValueError):
        sample_order(pools, sizes=(3, 2, 1))
    pools[2][0] = pools[0][0]
    with pytest.raises(ValueError):
        sample_order(pools, sizes=(2, 2, 1))


def test_new_and_retained_slices_remain_separate_and_missing_rows_fail():
    a, b = example(0), example(1, "color")
    c = dict(example(2), source="programmatic_decision_v1", task="numeric:count")
    assert row_key(a) == "relevance:count:q8:sentence"
    assert row_key(c) == "numeric:count"
    schema = {"relevance:count:q8:sentence": 2, "relevance:color:q8:sentence": 1}
    summary = metric_summary([a, dict(a, id="3"), b], np.array([[2, 0], [2, 0], [0, 2]]), schema)
    assert summary["overall"]["accuracy"] == pytest.approx(2 / 3)
    assert summary["macro"]["accuracy"] == 0.5
    assert summary["minimum_accuracy"] == 0
    with pytest.raises(ValueError):
        metric_summary([a, b], np.array([[2, 0], [0, 2]]), schema)
    with pytest.raises(ValueError):
        metric_summary(
            [a, b],
            np.array([[2, 0, 4], [0, 2, 4]]),
            {"relevance:count:q8:sentence": 1, "relevance:color:q8:sentence": 1},
        )


def test_accounting_does_not_count_repeated_questions_or_native_assets_as_unique():
    a = example(1)
    b = dict(example(2), state='{"count":"2"}')
    result = accounting([a, a, b])
    assert (result["consumed"], result["unique_questions"], result["unique_states"]) == (3, 2, 2)
    assert result["repeated_questions"] == 1 and result["native_assets"] == 0
    assert result["sources"] == {"condition_relevance_v1": 3}


def test_balanced_auxiliary_loss_has_correct_positive_negative_gradients():
    torch = pytest.importorskip("torch")
    scores = torch.zeros(4, requires_grad=True)
    loss = balanced_relevance_loss(scores, [True, False, False, False])
    assert float(loss.detach()) == pytest.approx(math.log(2))
    loss.backward()
    assert scores.grad.tolist() == pytest.approx([-0.25, 1 / 12, 1 / 12, 1 / 12])
    with pytest.raises(ValueError):
        balanced_relevance_loss(scores, [True] * 4)
    with pytest.raises(ValueError):
        balanced_relevance_loss(scores, [1, 0, 0, 0])


def test_native_forward_is_unchanged_and_auxiliary_gradient_reaches_shared_pooler():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    config = transformers.DebertaV2Config(
        vocab_size=20,
        hidden_size=12,
        num_hidden_layers=1,
        num_attention_heads=3,
        intermediate_size=24,
        pooler_hidden_size=12,
        num_labels=2,
    )
    model = transformers.DebertaV2ForSequenceClassification(config).eval()
    for p in model.deberta.embeddings.parameters():
        p.requires_grad_(False)
    head = torch.nn.Linear(12, 1)
    encoded = dict(
        input_ids=torch.tensor([[1, 2, 3], [1, 3, 4]]),
        attention_mask=torch.ones(2, 3, dtype=torch.long),
    )
    expected = model(**encoded).logits[:, 0].detach()
    score, relevance = forward_with_relevance(model, head, encoded)
    assert torch.equal(score.detach(), expected)
    balanced_relevance_loss(relevance, [True, False]).backward()
    assert model.pooler.dense.weight.grad.abs().sum() > 0
    assert head.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.deberta.embeddings.parameters())
    with torch.no_grad():
        head.weight.add_(100)
    actual, _ = forward_with_relevance(model, head, encoded)
    assert torch.equal(actual.detach(), expected)


def test_bootstrap_pairs_complete_state_clusters_and_rejects_missing_slice():
    rows = [example(f"{g}:{f}", f, g) for g in range(4) for f in ("count", "color")]
    wrong = np.tile([0.0, 1.0], (8, 1))
    right = np.tile([1.0, 0.0], (8, 1))
    result = bootstrap_relevance(rows, wrong, right, replicates=100)
    assert result["difference"] == 1
    assert result["interval95"] == [1.0, 1.0]
    assert result["states"] == 4
    assert bootstrap_relevance(rows, right, right, replicates=100)["interval95"] == [0.0, 0.0]
    with pytest.raises(ValueError):
        bootstrap_relevance(rows[1:], wrong[1:], right[1:], replicates=100)


def test_matched_loss_changes_only_auxiliary_coefficient_and_keeps_legacy_kl():
    torch = pytest.importorskip("torch")
    scores = torch.zeros(2, requires_grad=True)
    aux = torch.zeros(2, requires_grad=True)
    r = dict(example(1), supervision={"request_relevant": [True, False]})
    ce, info = training_objective(scores, aux, r, None, 0.0)
    total, other = training_objective(scores, aux, r, None, 0.1)
    assert float(ce.detach()) == pytest.approx(math.log(2))
    assert float(total.detach()) == pytest.approx(1.1 * math.log(2))
    assert info == other
    total.backward()
    assert scores.grad.tolist() == pytest.approx([-0.5, 0.5])
    assert aux.grad.tolist() == pytest.approx([-0.025, 0.025])
    legacy = dict(r, source="original")
    value, stats = training_objective(scores, aux, legacy, scores.detach(), 0.1)
    assert float(value.detach()) == pytest.approx(math.log(2))
    assert stats["kl"] == pytest.approx(0) and stats["relevance"] is None
    with pytest.raises(ValueError):
        training_objective(scores, aux, r, None, -1)


def test_checkpoint_restores_auxiliary_and_core_without_touching_frozen_weights(tmp_path):
    torch = pytest.importorskip("torch")
    model, head = torch.nn.Linear(2, 2), torch.nn.Linear(2, 1)
    model.bias.requires_grad_(False)
    frozen = model.bias.detach().clone()
    expected = model.weight.detach().clone()
    aux = head.weight.detach().clone()
    path = tmp_path / "delta.pt"
    save_trainable(model, head, ["weight"], path)
    with torch.no_grad():
        model.weight.add_(2)
        head.weight.add_(3)
    load_trainable(model, head, ["weight"], path)
    assert torch.equal(model.weight, expected) and torch.equal(head.weight, aux)
    assert torch.equal(model.bias, frozen)
    with pytest.raises(ValueError):
        load_trainable(model, head, ["weight", "bias"], path)
