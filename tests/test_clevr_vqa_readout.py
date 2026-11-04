import pytest

torch = pytest.importorskip("torch")


def test_linear_head_shape_and_gradients():
    from tiny_omni_decision.clevr_vqa_readout import ClevrAnswerReadout

    head = ClevrAnswerReadout()
    features = torch.randn(3, 768)
    output = head(features)
    assert output.shape == (3, 28)
    output.sum().backward()
    assert head.classifier.weight.grad is not None
    assert head.classifier.bias.grad is not None


def test_option_masked_loss_uses_only_supplied_answers():
    from tiny_omni_decision.clevr_vqa_options import ANSWER_LABELS
    from tiny_omni_decision.clevr_vqa_readout import (
        masked_option_logits,
        option_masked_cross_entropy,
        option_probabilities_batch,
    )

    answer_logits = torch.zeros(2, len(ANSWER_LABELS), requires_grad=True)
    answer_logits.data[0, ANSWER_LABELS.index("cyan")] = 2.0
    answer_logits.data[0, ANSWER_LABELS.index("red")] = 0.0
    candidates = torch.tensor(
        [
            [ANSWER_LABELS.index("red"), ANSWER_LABELS.index("cyan"), -1, -1],
            [ANSWER_LABELS.index("no"), ANSWER_LABELS.index("yes"), -1, -1],
        ]
    )
    counts = torch.tensor([2, 2])
    targets = torch.tensor([1, 0])

    logits = masked_option_logits(answer_logits, candidates, counts)
    assert logits.shape == (2, 4)
    assert torch.isneginf(logits[:, 2:]).all()
    probabilities = option_probabilities_batch(answer_logits, candidates, counts)
    torch.testing.assert_close(probabilities.sum(dim=-1), torch.ones(2))
    loss = option_masked_cross_entropy(answer_logits, candidates, counts, targets)
    assert torch.isfinite(loss)
    loss.backward()
    assert answer_logits.grad is not None
    assert answer_logits.grad[0, ANSWER_LABELS.index("cyan")] < 0


def test_option_mask_rejects_invalid_counts_indices_and_padding():
    from tiny_omni_decision.clevr_vqa_options import ANSWER_LABELS
    from tiny_omni_decision.clevr_vqa_readout import masked_option_logits

    logits = torch.zeros(1, len(ANSWER_LABELS))
    valid = ANSWER_LABELS.index("yes")
    with pytest.raises(ValueError, match="two to four"):
        masked_option_logits(logits, torch.tensor([[valid, -1, -1, -1]]), torch.tensor([1]))
    with pytest.raises(ValueError, match="outside"):
        masked_option_logits(
            logits,
            torch.tensor([[valid, len(ANSWER_LABELS), -1, -1]]),
            torch.tensor([2]),
        )
    with pytest.raises(ValueError, match="padded"):
        masked_option_logits(
            logits,
            torch.tensor([[valid, ANSWER_LABELS.index("no"), 0, -1]]),
            torch.tensor([2]),
        )


def test_quality_metrics_and_answer_kind_macro():
    from tiny_omni_decision.clevr_vqa_readout import macro_by_kind, quality_metrics

    probs = torch.tensor([[0.8, 0.2], [0.6, 0.4], [0.1, 0.9]])
    targets = torch.tensor([0, 1, 1])
    metrics = quality_metrics(probs, targets)
    assert metrics["count"] == 3
    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert metrics["nll"] > 0
    assert metrics["brier"] > 0
    macro = macro_by_kind(
        {"boolean": metrics, "count": {**metrics, "accuracy": 0.5}},
    )
    assert macro["accuracy"] == pytest.approx((2 / 3 + 0.5) / 2)
    assert macro["minimum_kind_accuracy"] == 0.5
