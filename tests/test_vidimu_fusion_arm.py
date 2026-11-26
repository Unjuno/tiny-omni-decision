import pytest

pytest.importorskip("torch")
import torch

from scripts.train_vidimu_fusion_arm import VidimuFusion


def test_wrapper_scores_thirteen_classes_and_masks_missing_video():
    torch.manual_seed(17)
    model = VidimuFusion()
    x = torch.randn(3, 6506)
    p = model(x).softmax(-1)
    assert p.shape == (3, 13) and torch.isfinite(p).all()
    reversed_logits = model(x, option_order=torch.arange(12, -1, -1))
    # Batched kernels and softmax reductions can round differently under permutation.
    torch.testing.assert_close(reversed_logits.flip(-1), model(x), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(reversed_logits.softmax(-1).flip(-1), p, rtol=1e-6, atol=1e-7)
    missing = model(x, missing="video")
    x[:, :6144] = float("nan")
    torch.testing.assert_close(missing, model(x, missing="video"), rtol=0, atol=0)
    with pytest.raises(ValueError, match="missing"):
        model(x, missing="unknown")
