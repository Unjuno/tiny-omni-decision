from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
decision_module = pytest.importorskip("tiny_omni_decision.xcapture_decision")
FrozenFeatureDecision = decision_module.FrozenFeatureDecision
decision_json = decision_module.decision_json
multiclass_metrics = decision_module.multiclass_metrics
paired_cross_modal_loss = decision_module.paired_cross_modal_loss
projection_parameter_count = decision_module.projection_parameter_count


def test_decision_head_produces_probabilities_and_option_order_is_preserved():
    torch.manual_seed(17)
    model = FrozenFeatureDecision().eval()
    query = torch.randn(1, 512)
    candidates = torch.randn(3, 384)
    original = model.option_probabilities("image", query, "audio", candidates)[0]
    permutation = torch.tensor([2, 0, 1])
    reordered = model.option_probabilities(
        "image", query, "audio", candidates[permutation]
    )[0]

    torch.testing.assert_close(reordered, original[permutation])
    assert int(reordered.argmax()) == int(torch.where(permutation == original.argmax())[0])


def test_decision_json_keeps_supplied_option_ids_and_normalized_probs():
    result = json.loads(decision_json(["b", "a", "c"], np.array([0.2, 0.7, 0.1])))

    assert result["predicted_option_id"] == "a"
    assert [row["option_id"] for row in result["probabilities"]] == ["b", "a", "c"]
    assert sum(row["probability"] for row in result["probabilities"]) == pytest.approx(1.0)


def test_decision_json_rejects_duplicate_options_and_bad_probabilities():
    with pytest.raises(ValueError, match="unique"):
        decision_json(["same", "same"], np.array([0.5, 0.5]))
    with pytest.raises(ValueError, match="sum to one"):
        decision_json(["a", "b"], np.array([0.2, 0.2]))


def test_metrics_include_confidence_one_in_ece():
    metrics = multiclass_metrics(torch.tensor([[100.0, 0.0], [100.0, 0.0]]))

    assert metrics["accuracy"] == 0.5
    assert metrics["ece"] == pytest.approx(0.5)


def test_metrics_accept_repeated_targets_for_bidirectional_retrieval():
    logits = torch.tensor(
        [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]] * 2
    )
    targets = torch.arange(3).repeat(2)

    metrics = multiclass_metrics(logits, targets=targets)

    assert metrics["accuracy"] == 1.0


def test_multimodal_contrastive_loss_reaches_each_projection():
    model = FrozenFeatureDecision()
    features = {
        "image": torch.randn(4, 2, 512),
        "audio": torch.randn(4, 2, 384),
        "tactile": torch.randn(4, 2, 512),
    }

    loss = paired_cross_modal_loss(model, features)
    loss.backward()

    assert torch.isfinite(loss)
    assert all(model.projections[name].weight.grad is not None for name in features)
    assert projection_parameter_count(model) == 180_225
