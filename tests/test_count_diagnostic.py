import numpy as np
import pytest

from tiny_omni_decision.count_diagnostic import count_errors


def test_count_confusion_signed_and_absolute_errors():
    r = count_errors([1, 2, 3, 3], [1, 3, 2, 1], classes=[1, 2, 3])
    assert r["exact_accuracy"] == 0.25
    assert r["underestimated"] == 2 and r["overestimated"] == 1
    assert r["mean_absolute_error"] == 1
    assert r["signed_error"] == -0.5
    assert r["confusion_true_by_predicted"] == [[1, 0, 0], [0, 0, 1], [1, 1, 0]]


def test_count_diagnostic_rejects_invalid_data():
    with pytest.raises(ValueError):
        count_errors([], [], classes=[1, 2])
    with pytest.raises(ValueError):
        count_errors([1], [3], classes=[1, 2])
    with pytest.raises(ValueError):
        count_errors([1.5], [1], classes=[1, 2])


def test_fixed_candidate_affine_equivalence():
    torch = pytest.importorskip("torch")
    from tiny_omni_decision.bilinear_readout import BilinearCandidateScorer
    from tiny_omni_decision.count_diagnostic import fixed_candidate_affine

    torch.manual_seed(17)
    model = BilinearCandidateScorer(8, 5, 4).double()
    candidates = torch.randn(3, 5, dtype=torch.float64)
    observations = torch.randn(7, 8, dtype=torch.float64)
    weight, bias = fixed_candidate_affine(model, candidates)
    inputs = torch.cat(
        (observations[:, None].expand(-1, 3, -1), candidates[None].expand(7, -1, -1)), -1
    )
    torch.testing.assert_close(
        model(inputs), observations @ weight.T + bias, rtol=1e-12, atol=1e-12
    )
    assert np.isfinite(weight.numpy()).all()


def test_grouped_loss_preserves_equal_task_weight_and_label_order():
    torch = pytest.importorskip("torch")
    from torch.nn import functional as functional

    from tiny_omni_decision.count_diagnostic import fixed_taxonomy_loss

    torch.manual_seed(17)
    logits = torch.randn(3, 4, 10)
    targets = torch.randint(10, (3, 4))
    expected = torch.stack([functional.cross_entropy(logits[:, i], targets[:, i])
                            for i in range(4)]).mean()
    torch.testing.assert_close(fixed_taxonomy_loss(logits, targets), expected)
    with pytest.raises(ValueError):
        fixed_taxonomy_loss(logits, targets.T)
