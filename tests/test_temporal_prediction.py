from importlib import import_module

import pytest

torch = pytest.importorskip("torch")
temporal_prediction = import_module("tiny_omni_decision.temporal_prediction")
NextFrameFeaturePredictor = temporal_prediction.NextFrameFeaturePredictor
cosine_prediction_loss = temporal_prediction.cosine_prediction_loss
per_sequence_cosine_error = temporal_prediction.per_sequence_cosine_error


def test_next_frame_predictor_is_causal_and_returns_one_prediction_per_transition():
    torch.manual_seed(17)
    model = NextFrameFeaturePredictor(feature_width=12, hidden_width=5)
    features = torch.randn(3, 8, 12)

    predictions = model(features)
    changed_future = features.clone()
    changed_future[:, 4:] = torch.randn_like(changed_future[:, 4:])
    future_changed_predictions = model(changed_future)
    loss = cosine_prediction_loss(predictions, features)

    assert predictions.shape == (3, 7, 12)
    assert torch.allclose(predictions[:, :4], future_changed_predictions[:, :4])
    assert torch.isfinite(predictions).all()
    assert torch.isfinite(loss)
    loss.backward()
    assert model.temporal.weight_ih_l0.grad is not None
    assert model.temporal.weight_ih_l0.grad.abs().sum() > 0


def test_per_sequence_score_is_finite_and_higher_for_wrong_temporal_order():
    features = torch.eye(4).reshape(1, 4, 4)
    # Predictions are aligned with the following frame, not the input frame.
    correct = features[:, 1:]
    wrong = correct.flip(1)

    correct_score = per_sequence_cosine_error(correct, features)
    wrong_score = per_sequence_cosine_error(wrong, features)

    assert correct_score.shape == (1,)
    assert torch.allclose(correct_score, torch.zeros_like(correct_score))
    assert wrong_score.item() > correct_score.item()


@pytest.mark.parametrize("width,hidden", [(0, 4), (4, 0), (True, 4), (4, 2.0)])
def test_predictor_rejects_invalid_dimensions(width, hidden):
    with pytest.raises(ValueError, match="positive integers"):
        NextFrameFeaturePredictor(feature_width=width, hidden_width=hidden)


def test_predictor_rejects_nonfinite_or_nonsequence_input():
    model = NextFrameFeaturePredictor(feature_width=3, hidden_width=2)

    with pytest.raises(ValueError, match="finite"):
        model(torch.zeros(2, 1, 3))
    with pytest.raises(ValueError, match="finite"):
        model(torch.tensor([[[float("nan"), 0.0, 1.0], [0.0, 1.0, 0.0]]]))


def test_prediction_loss_rejects_mismatched_or_nonfinite_inputs():
    with pytest.raises(ValueError, match="incompatible shapes"):
        cosine_prediction_loss(torch.zeros(2, 3, 4), torch.zeros(2, 5, 4))
    with pytest.raises(ValueError, match="incompatible shapes"):
        per_sequence_cosine_error(
            torch.full((1, 2, 3), float("inf")), torch.zeros(1, 3, 3)
        )
